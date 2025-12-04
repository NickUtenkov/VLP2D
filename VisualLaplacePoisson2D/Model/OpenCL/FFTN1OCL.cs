using Cloo;
using DD128Numeric;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FFTN1OCL<T> : IFFTOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		protected CommandQueueOCL commands;
		IFFTOCL<T> fft1;
		BufferOCL<T> lyambda1, lyambda2;
		KernelOCL kernel;
		long[] gWorkSize = { 0, 0 };

		public FFTN1OCL(CommandQueueOCL commands, IFFTOCL<T> fft1, int fftSize, int N2, T stepX2, T stepY2, bool useSineTransform)
		{
			this.commands = commands;
			this.fft1 = fft1;
			createLyambaArrays(fftSize, N2, stepX2, stepY2);
			createKernel((fftSize / 2 + 1) * 2);
			gWorkSize[1] = fftSize - 1;
		}

		public void calculate(BufferOCL<T> ioData, int workSize, T coef)
		{
			fft1.calculate(ioData, workSize, coef);
		}

		~FFTN1OCL()
		{
		}

		public void calculateDivideByLyambdasSum(BufferOCL<T> data, int workSize, int offsetJ)
		{
			gWorkSize[0] = workSize;
			kernel.SetMemoryArgument(0, data);
			kernel.SetValueArgument(4, offsetJ);
			commands.Execute(kernel, null, gWorkSize, null, null);

			commands.Finish();
		}

		void createLyambaArrays(int N1, int N2, T stepX2, T stepY2)
		{
			T[] lyambda1Tmp = calcLyambda(N1, stepX2);
			lyambda1 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, lyambda1Tmp);

			if ((N1 != N2) || (T.Abs(stepX2 - stepY2) > T.CreateTruncating(1E-10)))
			{
				T[] lyambda2Tmp = calcLyambda(N2, stepY2);
				lyambda2 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, lyambda2Tmp);
			}
			else
			{
				lyambda2 = lyambda1;
			}
		}

		T[] calcLyambda(int n, T step2)
		{
			T[] lyamba = new T[n - 1];
			T pi2N = T.Pi / T.CreateTruncating(n) / T.CreateTruncating(2);
			T _4 = T.CreateTruncating(4);
			for (int i = 1; i < n; i++)
			{
				T sin = T.Sin(pi2N * T.CreateTruncating(i));
				lyamba[i - 1] = (sin * sin * _4 / step2);
			}

			return lyamba;
		}

		void createKernel(int vectorLength)
		{
			string functionName = "lyambda";
			string args = string.Format("(global {0} *ioData, global {0} *lyambda1, global {0} *lyambda2, int vectorLength, int offsetJ)\n", Utils.getTypeName<T>());
			string srcLyambda =
@"
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	int idx = i * vectorLength + j + 1;
	ioData[idx] = HP(ioData[idx] / (lyambda1[j] + lyambda2[offsetJ + i]));
}
";
			string strProgram = UtilsCL.kernelPrefix + functionName + args + srcLyambda;
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}
			ProgramOCL program = UtilsCL.createProgram(strProgram, null, commands.Context, commands.Device);
			kernel = program.CreateKernel(functionName);
			kernel.SetMemoryArgument(1, lyambda1);
			kernel.SetMemoryArgument(2, lyambda2);
			kernel.SetValueArgument(3, vectorLength);
		}

		public void cleanup()
		{
			commands = null;
			if (lyambda2 != lyambda1) UtilsCL.disposeBuf(ref lyambda2);
			UtilsCL.disposeBuf(ref lyambda1);
			UtilsCL.disposeKP(ref kernel);
			fft1 = null;
		}
	}
}

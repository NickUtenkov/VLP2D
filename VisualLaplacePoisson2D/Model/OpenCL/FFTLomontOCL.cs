//#define CreateAll

using Cloo;
using DD128Numeric;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FFTLomontOCL<T> : FFTLomontBaseOCL<T>, IFFTOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CommandQueueOCL commands;
		int fftSize;//use in DEBUG(for printCUDABuffer)
		long[] gWorkSize = { 0 };

		public FFTLomontOCL(CommandQueueOCL commands, int fftSize) : base(commands, fftSize)
		{
			this.commands = commands;
			this.fftSize = fftSize;

			string functionNameReverse = fftSize > 2 ? "reverseFFT" : null;
			string functionNameTableFFT = "tableFFT";
			string functionNameRealFFT = "realFFT";
			string strDefines = createProgramDefines(fftSize);
			string compileOptionsNoOpt = typeof(T) == typeof(QD256) ? "-cl-opt-disable" : null;
			string compileOptionsOpt = null;

#if CreateAll
			string[] funcNames = [functionNameReverse, functionNameTableFFT, functionNameRealFFT];
			Func<string, string>[] funcs = [createProgramReverse, createProgramTableFFT, createProgramRealFFT];

			string strProgram = strDefines;
			for (int i = 0; i < funcNames.Length; i++) if (funcNames[i] != null) strProgram += funcs[i](funcNames[i]);

			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}

			ProgramOCL program = UtilsCL.createProgram(strProgram, compileOptionsOpt, commands.Context, commands.Device);
			ICollection<KernelOCL> kernels = program.CreateAllKernels();
			if (functionNameReverse != null) kernelReverse = kernels.First(x => x.FunctionName == functionNameReverse);
			kernelTableFFT = kernels.First(x => x.FunctionName == functionNameTableFFT);
			kernelRealFFT = kernels.First(x => x.FunctionName == functionNameRealFFT);
#else
			string strProgram;

			if (fftSize > 2)
			{
				strProgram = strDefines + createProgramReverse(functionNameReverse);
				kernelReverse = createKernel(strProgram, functionNameReverse, compileOptionsOpt);
			}

			strProgram = strDefines + createProgramTableFFT(functionNameTableFFT);
			kernelTableFFT = createKernel(strProgram, functionNameTableFFT, compileOptionsOpt);

			strProgram = strDefines + createProgramRealFFT(functionNameRealFFT);
			kernelRealFFT = createKernel(strProgram, functionNameRealFFT, compileOptionsOpt);//compileOptionsOpt cause InvalidCommandQueueException
#endif

			kernelReverse?.SetMemoryArgument(2, indJ);

			kernelTableFFT.SetMemoryArgument(2, sinTable);
			kernelTableFFT.SetMemoryArgument(3, cosTable);

			T theta = -T.Pi * T.CreateTruncating(2) / T.CreateTruncating(fftSize);
			kernelRealFFT.SetValueArgument(2, T.Cos(theta));
			kernelRealFFT.SetValueArgument(3, T.Sin(theta));
		}

		public void calculate(BufferOCL<T> data, int workSize, T coef)
		{
			sineTransform.preProcess(data, workSize);

			gWorkSize[0] = workSize;
			if (kernelReverse != null)
			{
				kernelReverse.SetMemoryArgument(0, data);
				kernelReverse.SetValueArgument(1, workSize);

				commands.Execute(kernelReverse, null, gWorkSize, null, null);
				commands.Finish();
			}

			kernelTableFFT.SetMemoryArgument(0, data);
			kernelTableFFT.SetValueArgument(1, workSize);
			commands.Execute(kernelTableFFT, null, gWorkSize, null, null);
			commands.Finish();

			kernelRealFFT.SetMemoryArgument(0, data);
			kernelRealFFT.SetValueArgument(1, workSize);
			commands.Execute(kernelRealFFT, null, gWorkSize, null, null);
			commands.Finish();

			sineTransform.postProcess(data, workSize, coef);
		}

		public void calculateDivideByLyambdasSum(BufferOCL<T> ioData, int workSize, int offset)
		{
		}

		public new void cleanup()
		{
			base.cleanup();

			UtilsCL.disposeKP(ref kernelReverse);
			UtilsCL.disposeKP(ref kernelTableFFT);
			UtilsCL.disposeKP(ref kernelRealFFT);
		}

		KernelOCL createKernel(string strProgram, string functionName, string compileOptions)
		{
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}

			ProgramOCL program = UtilsCL.createProgram(strProgram, compileOptions, commands.Context, commands.Device);
			return program.CreateKernel(functionName);
		}
	}
}

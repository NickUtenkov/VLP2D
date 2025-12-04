
using Cloo;
using DD128Numeric;
using QD256Numeric;
using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class VariablesSeparationProgonkaOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, IMinMaxValue<T>, ILogarithmicFunctions<T>
	{
		CommandQueueOCL commands;
		KernelOCL kernelProgonka;
		long[] gWorkSizeProgonka = { 0 };
		T[,] un;
		BufferOCL<T> unOCL, alfa;
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		int[] alfaOffsets, alfaCounts;
		BufferOCL<int> alfaOffsetsOCL, alfaCountsOCL;

		public VariablesSeparationProgonkaOCL(CommandQueueOCL commands, BufferOCL<T> unOCL, T[,] un, int progonkaSize, int N2, T stepX2, T stepY2)
		{
			if ((progonkaSize & 1) == 1) throw new System.Exception("VariablesSeparationProgonkaOCL progonkaSize should be even");

			this.commands = commands;
			this.unOCL = unOCL;
			this.un = un;

			int dim2 = N2 - 1;
			alfaOffsets = new int[dim2];
			alfaCounts = new int[dim2];
			T _PiN2 = T.Pi / T.CreateTruncating(N2);
			int alfaAllCount0 = calcAlphaOffsetAndSize(_PiN2, stepX2 / stepY2, progonkaSize / 2 - 1);
			int alfaAllCount = progonkaSize * dim2;
			alfa = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, alfaAllCount);
			alfaOffsetsOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, alfaOffsets);
			alfaCountsOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, alfaCounts);

			kernelProgonka = createKernelMeetingProgonka(progonkaSize);
			object[] args = [unOCL, alfa, alfaOffsetsOCL, alfaCountsOCL, stepX2, _PiN2, stepX2 / stepY2];
			UtilsCL.setKernelArguments<T>(kernelProgonka, args);
		}

		~VariablesSeparationProgonkaOCL()
		{
		}

		public void calculate(int[] stripWidths, Action<float> showProgress, float progressPercent, Func<bool> areIterationsCanceled)
		{
			int offsetJ = 0, dim1 = un.GetUpperBound(0) + 1, dim2 = un.GetUpperBound(1) + 1;
			SysIntX2 offsSrc = new SysIntX2(0, 0), offsDst = new SysIntX2(0, 0), region = new SysIntX2(stripWidths[0], dim1);
			int sizeT = Marshal.SizeOf(typeof(T));

			for (int i = 0; i < stripWidths.Length; i++)
			{
				region.X = (IntPtr)stripWidths[i];

				commands.WriteToBuffer(un, unOCL, true, offsSrc, offsDst, region, stripWidths[i] * sizeT, dim2 * sizeT, null);//sourceRowPitch & destinationRowPitch are wrong interchanged
				//Utils.printArray(un, "data host", "{0:0.000}", 9);
				//UtilsCL.printOCLBuffer(unOCL, commands, stripWidths[i], dim2, "data OCL");
				calculateStrip(stripWidths[i], offsetJ);
				commands.ReadFromBuffer(unOCL, ref un, true, offsDst, offsSrc, region, stripWidths[i] * sizeT, dim2 * sizeT, null);//sourceRowPitch & destinationRowPitch are wrong interchanged
				//UtilsCL.printOCLBuffer(unOCL, commands, stripWidths[i], dim2, "calculateStrip OCL");
				//Utils.printArray(un, "calculateStrip host", "{0:0.000}", 9);

				offsSrc.X += stripWidths[i];
				offsetJ += stripWidths[i];

				if (areIterationsCanceled()) return;
				showProgress(progressPercent / stripWidths.Length);
			}
		}

		public void cleanup()
		{
			commands = null;
			UtilsCL.disposeKP(ref kernelProgonka);
			kernelProgonka = null;
			unOCL = null;
			un = null;
			alfa?.Dispose();
			alfa = null;
			alfaOffsetsOCL?.Dispose();
			alfaOffsetsOCL = null;
			alfaCountsOCL?.Dispose();
			alfaCountsOCL = null;
		}

		KernelOCL createKernelMeetingProgonka(int progonkaSize)
		{
			string definesProgonka =
@"
#define U		{0}//upper bound
#define midX	{1}
#define RHS(j) HP(stepX2 * un[(j)])
#define ind(a) (((a) < alfaCounts[j + offsetJ]) ? (a) : alfaCounts[j + offsetJ] - 1)
#define identity(a) (a)
";
			string functionName = "progonkaMeeting";
			string srcProgonka =
	@"
(global {0} *un, global {0} *alfaIn, global int *alfaOffsets, global int *alfaCounts, {0} stepX2, {0} pi2N2, {0} mult, int offsetJ)
{{
	int j = get_global_id(0);
	int dimY = get_global_size(0);

	global {0} *alfa = alfaIn + alfaOffsets[j + offsetJ];

	//calc alfa
	{0} diagElem = HP(2.0 + mult * (2.0 - 2.0 * cos({1})));
	alfa[0] = HP(1.0 / diagElem);//[SNR] p.75(7)
	for (int i = 1; i < alfaCounts[j + offsetJ]; i++) alfa[i] = HP(1.0 / (diagElem - alfa[i - 1]));//[SNR] p.75(7)

	//calc beta, put to un
	//from left to middle, than from right to middle
	for (int i = 0; i <= midX - 1; i++)
	{{
		un[(i + 0) * dimY + j] = HP((RHS((i + 0) * dimY + j) + identity(i != 0 ? un[(i - 1) * dimY + j] : Zero)) * alfa[ind(i)]);
		un[(U - i) * dimY + j] = HP((RHS((U - i) * dimY + j) + identity(i != 0 ? un[(U - (i - 1)) * dimY + j] : Zero)) * alfa[ind(i)]);
	}}

	un[midX * dimY + j] = HP((un[midX * dimY + j] + alfa[ind(midX - 1)] * un[(midX - 1) * dimY + j]) / (1.0 - (alfa[ind(midX - 1)] * alfa[ind(midX - 1)])));

	//from middle to left, than from middle to right
	for (int i = midX - 1,i1 = i * dimY; i >= 0; i--,i1 -= dimY) un[i1 + j] = HP(un[i1 + j] + alfa[ind(i)] * un[i1 + dimY + j]);//[SNR] p.75(7)
	for (int i = midX + 1,i1 = i * dimY; i <= U; i++,i1 += dimY) un[i1 + j] = HP(un[i1 + j] + alfa[ind(U - i)] * un[i1 - dimY + j]);//[SNR] p.75(7)
}}
";
			string defines = string.Format(definesProgonka, progonkaSize - 1, progonkaSize / 2);
			string strProgram = defines + UtilsCL.kernelPrefix + functionName + string.Format(srcProgonka, Utils.getTypeName<T>(), ArithmeticReplacer.convertTo_mul_HD<T>("pi2N2", "(j + offsetJ + 1)"));
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + ArithmeticReplacer.replaceHPMacros(HighPrecisionOCL.strDD128Trig) + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + ArithmeticReplacer.replaceHPMacros(HighPrecisionOCL.strQD256Trig) + strProgram;
			}

			string compileOptions = typeof(T) == typeof(QD256) ? "-cl-opt-disable" : null;//to not cause InvalidCommandQueueException
			ProgramOCL programProgonka = UtilsCL.createProgram(strProgram, compileOptions, commands.Context, commands.Device);
			return programProgonka.CreateKernel(functionName);
		}

		void calculateStrip(int workSize, int offsetJ)
		{
			try
			{
				kernelProgonka.SetValueArgument(7, offsetJ);
				gWorkSizeProgonka[0] = workSize;
				commands.Execute(kernelProgonka, null, gWorkSizeProgonka, null, null);
				commands.Finish();
			}
			catch (Exception ex)
			{
				Debug.WriteLine(ex.Message);
			}
		}

		int calcAlphaOffsetAndSize(T pi2N2, T mult, int upper)
		{
			int offset = 0;
			for (int j = 0; j < alfaCounts.Length; j++)
			{
				T cosinus = T.Cos(pi2N2 * T.CreateTruncating(j + 1));
				T diagElem = (T.One + mult * (T.One - cosinus)) * T.CreateTruncating(2);
				int kUp = αCC.upperBound(diagElem, upper);
				alfaCounts[j] = kUp + 1;
				alfaOffsets[j] = offset;
				offset += alfaCounts[j];
			}
			return offset;
		}
	}
}

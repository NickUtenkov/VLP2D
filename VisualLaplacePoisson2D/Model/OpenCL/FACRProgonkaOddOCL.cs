
using Cloo;
using DD128Numeric;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class FACRProgonkaOddOCL<T> where T : struct, INumber<T>, IRootFunctions<T>
	{
		CommandQueueOCL commands;
		KernelOCL kernel;
		BufferOCL<T> alphaOCL;
		T[] alfa;
		long[] workSize = { 0 };
		int N2, L, alfaSize;
		T x2DivY2;
		BufferOCL<T> unOCL;
		int[] alfaOffsets, alfaCounts;
		BufferOCL<int> alfaOffsetsOCL, alfaCountsOCL;
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		T _2 = T.CreateTruncating(2);

		public FACRProgonkaOddOCL(CommandQueueOCL commands, BufferOCL<T> unOCL, int progonkaSize, int N2, int paramL, T stepX2, T stepY2)
		{
			if ((progonkaSize & 1) == 1) throw new System.Exception("FACRProgonkaOddOCL progonkaSize should be even");
			this.commands = commands;
			this.N2 = N2;
			L = paramL;
			this.unOCL = unOCL;
			alfaSize = progonkaSize / 2;
			x2DivY2 = stepX2 / stepY2;

			int maxDiagElems = 1 << (L - 1);
			alfaOffsets = new int[maxDiagElems];
			alfaCounts = new int[maxDiagElems];
			alfaOffsetsOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadWrite, maxDiagElems);
			alfaCountsOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadWrite, maxDiagElems);

			int maxAlfa = maxAlfaArrayElements();//instead of alfaSize * maxDiagElems
			int maxAlfaOld = alfaSize * maxDiagElems;
			alfa = new T[maxAlfa];
			alphaOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, maxAlfa);

			kernel = createKernelMeetingProgonkaOdd(progonkaSize, N2 - 1);
			object[] args = [unOCL, alphaOCL, alfaOffsetsOCL, alfaCountsOCL, x2DivY2, stepX2, 0, 0];
			UtilsCL.setKernelArguments<T>(kernel, args);
		}

		public void calculate(Func<bool> areIterationsCanceled, int dim1, T[] unShow, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			for (int l = L; l >= 1; l--)
			{
				calcAlphaOCL(l);

				kernel.SetValueArgument(6, l);
				kernel.SetValueArgument(7, 1 << (l - 1));//idxDelta

				workSize[0] = N2 >> l;
				commands.Execute(kernel, null, workSize, null, null);
				commands.Finish();
				if (areIterationsCanceled()) return;
				if (unShow != null)
				{
					int dim2 = N2 - 1;
					for (int jLoop = 1; jLoop <= workSize[0]; jLoop++)
					{
						int jl = (jLoop << l) - (1 << (l - 1)) - 1;//2,6,10,14,...(for L==2)
						for (int i = 0; i < dim1; i++) commands.ReadFromBuffer(unOCL, ref unShow, true, i * dim2 + jl, i * dim2 + jl, 1, null);// unShow[i * dim2 + jl] = unOCL[i * dim2 + jl];
					}
					UtilsPict.addPicture(lstBitmap, true, null, new Adapter2D<float>(dim1, dim2, (m, k) => float.CreateTruncating(unShow[m * dim2 + k])), fCreateBitmap);
				}
			}
		}

		public void cleanup()
		{
			commands = null;
			UtilsCL.disposeKP(ref kernel);
			UtilsCL.disposeBuf(ref alphaOCL);
			UtilsCL.disposeBuf(ref alfaOffsetsOCL);
			UtilsCL.disposeBuf(ref alfaCountsOCL);
			unOCL = null;
		}

		KernelOCL createKernelMeetingProgonkaOdd(int progonkaSize, int dimY)
		{
			string definesProgonka =
@"
#define U	{0}
#define midX	{1}
#define dimY	{2}
#define RHS1(j, k) HP(stepX2 * un[j] + x2DivY2 * (edgeLeft(j, k,idxDelta, un) + edgeRight(j, k,idxDelta, un)))
#define RHS2(j) HP(x2DivY2 * un[(j)])
//#define RHS(c, a, b)	(c ? RHS1(a, b) : RHS2(a))
#define identity(a)	(a)
#define ind(i, k) (((i) < alfaCounts[k]) ? (i) : alfaCounts[k] - 1)
#define indU(i)	(i) * dimY + (j)

";
			string strFunctions = string.Format(
@"
__attribute__((always_inline))
{0} edgeLeft(int j, int k, int idxDelta, global {0} *un)
{{
	return k - idxDelta > 0 ? un[j - idxDelta] : Zero;
}}

__attribute__((always_inline))
{0} edgeRight(int j, int k, int idxDelta, global {0} *un)
{{
	return k + idxDelta < dimY - 1 ? un[j + idxDelta] : Zero;
}}
", Utils.getTypeName<T>());

			string functionName = "progonkaMeetingM2Odd";
			string srcProgonka =
	@"(global {2} *un, global {2} *alfa, global int *alfaOffsets, global int *alfaCounts, {2} x2DivY2, {2} stepX2, int l, int idxDelta)
{{
	int j = ((get_global_id(0) + 1) << l) - idxDelta - 1;//odd in CPU variant(because of boundaries)

{0}
{1}
}}
";
			string srcProgonkaL0 =
	@"
	//calc beta, put to un
	//from left to middle & from right to middle
	for (int i = 0; i <= midX - 1; i++)
	{{
		un[indU(i + 0)] = HP((RHS1(indU(i + 0), j) + identity(i != 0 ? un[indU(i - 1)] : Zero)) * alfa[ind(i, 0)]);
		un[indU(U - i)] = HP((RHS1(indU(U - i), j) + identity(i != 0 ? un[indU(U - (i - 1))] : Zero)) * alfa[ind(i, 0)]);
	}}

	un[indU(midX)] = HP((un[indU(midX)] + alfa[ind(midX - 1, 0)] * un[indU(midX - 1)]) / (1.0 - alfa[ind(midX - 1, 0)] * alfa[ind(midX - 1, 0)]));//[SNR] p.77(bottom)

	//from middle to left, than from middle to right
	for (int i = midX - 1,i1 = i * dimY; i >= 0; i--,i1 -= dimY) un[i1 + j] = HP(un[i1 + j] + alfa[ind(i, 0)] * un[i1 + dimY + j]);
	for (int i = midX + 1,i1 = i * dimY; i <= U; i++,i1 += dimY) un[i1 + j] = HP(un[i1 + j] + alfa[ind(U - i, 0)] * un[i1 - dimY + j]);
";
			string srcProgonkaLoop = string.Format(
	@"
	for (int k = 1; k < 1 << (l - 1); k++)
	{{
		global {0} *alfa2 = alfa + alfaOffsets[k];
		//calc beta, put to un
		//from left to middle & from right to middle
		for (int i = 0; i <= midX - 1; i++)
		{{
			un[indU(i + 0)] = HP((RHS2(indU(i + 0)) + identity(i != 0 ? un[indU(i - 1)] : Zero)) * alfa2[ind(i, k)]);
			un[indU(U - i)] = HP((RHS2(indU(U - i)) + identity(i != 0 ? un[indU(U - (i - 1))] : Zero)) * alfa2[ind(i, k)]);
		}}

		un[indU(midX)] = HP((un[indU(midX)] + alfa2[ind(midX - 1, k)] * un[indU(midX - 1)]) / (1.0 - alfa2[ind(midX - 1, k)] * alfa2[ind(midX - 1, k)]));//[SNR] p.77(bottom)

		//from middle to left, than from middle to right
		for (int i = midX - 1,i1 = i * dimY; i >= 0; i--,i1 -= dimY) un[i1 + j] = HP(un[i1 + j] + alfa2[ind(i, k)] * un[i1 + dimY + j]);
		for (int i = midX + 1,i1 = i * dimY; i <= U; i++,i1 += dimY) un[i1 + j] = HP(un[i1 + j] + alfa2[ind(U - i, k)] * un[i1 - dimY + j]);
	}}
", Utils.getTypeName<T>());
			string defines = string.Format(definesProgonka, progonkaSize - 1, progonkaSize / 2, dimY);
			string strProgonka = string.Format(UtilsCL.kernelPrefix + functionName + srcProgonka, srcProgonkaL0, (L > 1) ? srcProgonkaLoop : "", Utils.getTypeName<T>());
			string strProgram = defines + strFunctions + strProgonka;
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}

			string compileOptions = typeof(T) == typeof(QD256) ? "-cl-opt-disable" : null;//to not cause InvalidCommandQueueException
			ProgramOCL program = UtilsCL.createProgram(strProgram, compileOptions, commands.Context, commands.Device);
			return program.CreateKernel(functionName);
		}

		void calcAlphaOCL(int curL)
		{
			void calcAlpha(T diagElem, int offset, int alfaCount)
			{
				alfa[0 + offset] = T.One / diagElem;//[SNR] p.195(40)
				for (int i = 1; i < alfaCount; i++) alfa[i + offset] = T.One / (diagElem - alfa[i - 1 + offset]);//[SNR] p.195(40)
			}
			T[] diag = new T[1 << (curL - 1)];
			Utils.generateSqrtCoefs<T>(curL - 1, (i, val) => diag[i] = val + _2);
			int offs = 0;
			for (int i = 0; i < diag.Length; i++)
			{
				T diagElem = x2DivY2 * diag[i] + _2;
				int kUp = αCC.upperBound(diagElem, alfaSize - 1) + 1;
				calcAlpha(diagElem, offs, kUp);
				alfaOffsets[i] = offs;
				alfaCounts[i] = kUp;
				offs += kUp;
			}
			commands.WriteToBuffer(alfa, alphaOCL, true, 0, 0, offs, null);
			commands.WriteToBuffer(alfaOffsets, alfaOffsetsOCL, true, 0, 0, diag.Length, null);
			commands.WriteToBuffer(alfaCounts, alfaCountsOCL, true, 0, 0, diag.Length, null);
		}

		int maxAlfaArrayElements()
		{
			int maxDiagElems = 1 << (L - 1);
			T[] diag = new T[maxDiagElems];
			int maxAlfa = 0;
			for (int l = L; l >= 1; l--)
			{
				int kAll = 0;
				Utils.generateSqrtCoefs<T>(l - 1, (i, val) => diag[i] = val + _2);
				for (int i = 0; i < diag.Length; i++)
				{
					T diagElem = x2DivY2 * diag[i] + _2;
					int kUp = αCC.upperBound(diagElem, alfaSize - 1) + 1;
					kAll += kUp;
				}
				maxAlfa = Math.Max(maxAlfa, kAll);
			}
			return maxAlfa;
		}
	}
}

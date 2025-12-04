using Cloo;
using DD128Numeric;
using QD256Numeric;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class FACRForwardStepsOCL<T> where T : struct, INumber<T>, IRootFunctions<T>
	{
		CommandQueueOCL commands;
		KernelOCL kernel;
		long[] gWorkSize = { 0 };
		BufferOCL<T> multiplied, accum, coef;
		int M2, L;
		T diagElem;

		public FACRForwardStepsOCL(CommandQueueOCL commands, BufferOCL<T> unOCL, int dim1, int dim2, int N2, int valueL, T hYX2)
		{
			this.commands = commands;

			M2 = N2 >> 1;
			int maxWorkSize = M2 - 1;
			L = valueL;
			try
			{
				multiplied = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dim1 * maxWorkSize);
				accum = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dim1 * maxWorkSize);
				coef = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, 1 << (L - 1));
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			createKernel(commands.Device, commands.Context, unOCL, dim1, dim2, hYX2);

			diagElem = (T.One + hYX2) * T.CreateTruncating(2.0);
		}

		public void calculate(Func<bool> areIterationsCanceled)
		{
			int m = M2;

			T[] diag = new T[1 << (L - 1)];
			for (int l = 1; l <= L; l++)
			{
				Utils.generateSqrtCoefs<T>(l - 1, (i, val) => diag[i] = (diagElem + val));
				int cElems = 1 << (l - 1);
				commands.WriteToBuffer(diag, coef, true, 0, 0, cElems, null);

				kernel.SetValueArgument(6, l);
				kernel.SetValueArgument(7, cElems);
				gWorkSize[0] = m - 1;
				commands.Execute(kernel, null, gWorkSize, null, null);

				m >>= 1;
				if (areIterationsCanceled()) return;
			}
		}

		public void cleanup()
		{
			commands = null;
			UtilsCL.disposeKP(ref kernel);
			UtilsCL.disposeBuf(ref multiplied);
			UtilsCL.disposeBuf(ref accum);
			UtilsCL.disposeBuf(ref coef);
		}

		void createKernel(DeviceOCL device, ContextOCL context, BufferOCL<T> unOCL, int dim1, int dim2, T hYX2)
		{
			string definesKernelFormat =
@"
#define ub1	{0}//upper bound on 1st dimension
#define fn(i,j) fn[(i) * {1} + (j)]
#define sum(i,j,n) HP(fn(i, j - n) + fn(i, j + n))//may be danger without braces

";
			string srcMult = string.Format(
@"global {0} *matricesCMultipleVector(int colFn, global {0} *fn, global {0} *src, global {0} *dst, {0} hYX2, int cMatrices, global {0} *diag)
{{
	for (int j = 0; j <= ub1; j++) src[j] = fn(j, colFn);

	for (int k = 0; k < cMatrices; k++)
	{{
		dst[0] = HP(diag[k] * src[0] - hYX2 * (0.0 + src[1]));//0.0f == src[-1], which doesn't exist
		for (int j = 1; j < ub1; j++)
		{{
			dst[j] = HP(diag[k] * src[j] - hYX2 * (src[j - 1] + src[j + 1]));
		}}
		dst[ub1] = HP(diag[k] * src[ub1] - hYX2 * (src[ub1 - 1] + 0.0));//0.0f == src[ub1 + 1], which doesn't exist

		global {0} *tmp = src;
		src = dst;
		dst = tmp;
	}}
	return src;
}}
", Utils.getTypeName<T>());
			string functionName = "forwardLSteps";
			string srcKernel = string.Format(
@"(global {0} *fn, global {0} *multiplied, global {0} *accum, global {0} *diag, {0} hYX2, int dim1, int l, int n)
{{
	int row = get_global_id(0);
	int j = ((row + 1) << l) - 1;

	int offs = row * dim1;
	global {0} *res = matricesCMultipleVector(j, fn, multiplied + offs, accum + offs, hYX2, n, diag);
	{0} prev = HP(sum(0, j, n) + res[0]);
	for (int i = 1; i < ub1; i++)
	{{
		{0} cur = HP(sum(i, j, n) + res[i]);//[SNR] p.202, (19) and p.201, (15)
		fn(i - 1, j) = prev;
		prev = cur;
	}}
	fn(ub1 - 1, j) = prev;
	fn(ub1, j) = HP(sum(ub1, j, n) + res[ub1]);
}}", Utils.getTypeName<T>());
			string definesKernel = string.Format(definesKernelFormat, dim1 - 1, dim2);
			string strProgram = definesKernel + srcMult + UtilsCL.kernelPrefix + functionName + srcKernel;
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, context, device);
			kernel = program.CreateKernel(functionName);
			kernel.SetMemoryArgument(0, unOCL);
			kernel.SetMemoryArgument(1, multiplied);
			kernel.SetMemoryArgument(2, accum);
			kernel.SetMemoryArgument(3, coef);
			kernel.SetValueArgument(4, hYX2);
			kernel.SetValueArgument(5, dim1);
		}
	}
}

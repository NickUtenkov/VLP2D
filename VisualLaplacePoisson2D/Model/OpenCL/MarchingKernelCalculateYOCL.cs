using Cloo;
using DD128Numeric;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelCalculateYOCL<T>
	{
		public static KernelOCL createKernelCalculateY(DeviceOCL device, ContextOCL context, int L, int k, int Nx, int M, int colsUn, int colsFn, bool ksiIsNull)
		{
			string functionName = "calculateY";
			string args = string.Format("(global {0}* __restrict__ un, const global {0}* __restrict__ fn, {0} cBase, {0} ai, {0} bi)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string defines =
@"
#define L				{0}
#define k				{1}
#define Nx				{2}
#define M				{3}
#define colsUn			{4}
#define colsFn			{5}
#define ksiIsNull		{6}
#define upperL	(L * 2)

";
			string functions = string.Format(
@"
void matrixCMultiplyArrayVector(global {0} *_dst, int rowDst, int rowSrc, {0} cBase, {0} ai, {0} bi)
{{
	global {0} *dst = _dst + rowDst * colsUn;
	global {0} *src = _dst + rowSrc * colsUn;
	dst[1] = HP(cBase * src[1] - bi * src[2]);
	for (int i = 2; i < M; i++)
	{{
		dst[i] = HP(-ai * src[i - 1] + cBase * src[i] - bi * src[i + 1]);//negation
	}}
	dst[M] = HP(-ai * src[M - 1] + cBase * src[M]);//negation
}}

void arrayRowSubtractRow(global {0} *_dst, int rowDst, int rowSrc)
{{
	global {0} *dst = _dst + rowDst * colsUn;
	global {0} *src = _dst + rowSrc * colsUn;
	for (int j = 1; j <= M; j++) dst[j] = HP(dst[j] - src[j]);
}}

void arrayRowSubtractVector(global {0}* __restrict__ _dst, int rowDst, const global {0}* __restrict__ src)
{{
	global {0} *dst = _dst + rowDst * colsUn;
	for (int j = 1; j <= M; j++) dst[j] = HP(dst[j] - src[j]);
}}

void arrayRowSubtractShortVector(global {0}* __restrict__ _dst, int rowDst, const global {0}* __restrict__ src)
{{
	global {0} *dst = _dst + rowDst * colsUn;
	dst[1] = HP(dst[1] - src[1]);
	dst[M] = HP(dst[M] - src[M]);
}}

void calculateYFromLeftOrRight(int l, global {0}* __restrict__ un, const global {0}* __restrict__ fn, {0} cBase, {0} ai, {0} bi)//SNE p.232(30)
{{
	bool left = (l < L);
	l = (l >= L) ? l - L : l;
	int idxStart = left ? 2 * l * k + 2 : (2 * l + 1) * k + 1;
	int idxEnd = left ? (2 * l + 1) * k : 2 * (l + 1) * k - 1;
	for (int idx = idxStart; idx <= idxEnd; idx++)
	{{
		int i = left ? idx : (idxStart + idxEnd) - idx;
		int i1 = left ? i - 1 : i + 1;
		int i2 = left ? i - 2 : i + 2;
		matrixCMultiplyArrayVector(un, i, i1, cBase, ai, bi);

		bool condSubtraction1 = left ? i - 2 > 0 : i + 2 < Nx;
		if (condSubtraction1) arrayRowSubtractRow(un, i, i2);//skip subtracting left/right edge column which assumed to be zero

		bool condSubtraction2 = left ? ksiIsNull && i - 1 > 1 : ksiIsNull && i + 1 < Nx - 1;
		condSubtraction2 ? arrayRowSubtractShortVector(un, i, fn + (i1 - 1) * colsFn) : arrayRowSubtractVector(un, i, fn + (i1 - 1) * colsFn);//'- 1' because dim1Fn == Nx - 1(not Nx + 1)
	}}
}}
", Utils.getTypeName<T>());
			string programSource =
@"
{
	int l = get_global_id(0);

	if (l < upperL)
	{
		calculateYFromLeftOrRight(l, un, fn, cBase, ai, bi);
	}
}";
			string strDefines = string.Format(defines, L, k, Nx, M, colsUn, colsFn, ksiIsNull ? "true" : "false");
			string strProgram = strDefines + functions + strProgramHeader + programSource;
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, context, device);
			return program.CreateKernel(functionName);
		}
	}
}


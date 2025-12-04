using Cloo;
using DD128Numeric;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelCalculatePQVectorsOCL<T> where T : struct, INumber<T>
	{
		public static KernelOCL createKernelCalculatePQVectors(DeviceOCL device, ContextOCL context, int L, int k, int Nx, int M, int colsPQ, int colsFn, bool ksiIsNull)
		{
			string functionName = "calculatePQVectors";
			string args = string.Format("(global {0}* __restrict__ pq0, global {0}* __restrict__ pq1, global {0}* __restrict__ pq2, global const {0}* __restrict__ fn, {0} cBase, {0} ai, {0} bi)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string defines =
@"
#define L				{0}
#define k				{1}
#define Nx				{2}
#define M				{3}
#define colsPQ			{4}
#define colsFn			{5}
#define ksiIsNull		{6}
#define upperL	(L * 2)

";
			string functions = string.Format(
@"
int lowerPartOffset(int offs, int j)
{{
    return offs + j;
}}

int upperPartOffset(int offs, int j)
{{
    return offs + 1 - j;
}}

void vectorAssignValue(global {0} *dst, {0} val)
{{
	for (int i = 0; i < M; i++) dst[i] = val;
}}

void vectorAssignRow(global {0}* __restrict__ dst, global const {0}* __restrict__ src)
{{
	for (int i = 0; i < M; i++) dst[i] = src[i + 1];
}}

void vectorAssignShortRow(global {0}* __restrict__ dst, global const {0}* __restrict__ src)
{{
	dst[0] = src[1];
	dst[M - 1] = src[M];
}}

void vectorAddRow(global {0}* __restrict__ dst, global const {0}* __restrict__ src)
{{
	for (int i = 0; i < M; i++) dst[i] = HP(dst[i] + src[i + 1]);
}}

void vectorAddShortRow(global {0}* __restrict__ dst, global const {0}* __restrict__ src)
{{
	dst[0] = HP(dst[0] + src[1]);
	dst[M - 1] = HP(dst[M - 1] + src[M]);
}}

void matrixCMultiplyVector(global {0}* __restrict__ dst, global {0}* __restrict__ src, {0} cBase, {0} ai, {0} bi)
{{
	dst[0] = HP(cBase * src[0] - bi * src[1]);
	for (int i = 1; i < M - 1; i++)
	{{
		dst[i] = HP(cBase * src[i] - ai * src[i - 1] - bi * src[i + 1]);
	}}
	dst[M - 1] = HP(cBase * src[M - 1] - ai * src[M - 2]);
}}

void vectorSubtractVector(global {0}* __restrict__ dst, global {0}* __restrict__ src)
{{
	for (int i = 0; i < M; i++) dst[i] = HP(dst[i] - src[i]);
}}

void initCalculatePQ(global {0}* __restrict__ pq1, global {0}* __restrict__ pq2, global const {0}* __restrict__ fn, bool condition)
{{
	vectorAssignValue(pq1, Zero);
	(ksiIsNull && condition) ? vectorAssignShortRow(pq2, fn) : vectorAssignRow(pq2, fn);
}}

void calculatePQ(global {0}* __restrict__ *pq0, global {0}* __restrict__ *pq1, global {0}* __restrict__ *pq2, global const {0}* __restrict__ fn, {0} cBase, {0} ai, {0} bi)
{{
	global {0}* __restrict__ tmp = *pq0;
	*pq0 = *pq1;
	*pq1 = *pq2;
	*pq2 = tmp;

	matrixCMultiplyVector(*pq2, *pq1, cBase, ai, bi);
	vectorSubtractVector(*pq2, *pq0);
	ksiIsNull ? vectorAddShortRow(*pq2, fn) : vectorAddRow(*pq2, fn);
}}
", Utils.getTypeName<T>());

			string programSource = string.Format(
@"
{{
	int l = get_global_id(0);

	if (l < upperL)
	{{
		bool lowerPart = (l < L);
		int lForOffs = (l >= L) ? l - L : l;
		int offs = lowerPart ? 2 * lForOffs * k : 2 * (lForOffs + 1) * k;
		bool cond = lowerPart ? offs > 1 : offs < Nx - 1;
		int offsFn = lowerPart ? offs + 1 : offs;

		global {0} *ptrPQ0 = pq0 + l * colsPQ;
		global {0} *ptrPQ1 = pq1 + l * colsPQ;
		global {0} *ptrPQ2 = pq2 + l * colsPQ;

		initCalculatePQ(ptrPQ1, ptrPQ2, fn + (offsFn - 1) * colsFn, cond);//' - 1' because dim1Fn == Nx - 1(not Nx + 1)

		for (int j = 2; j <= k; j++) calculatePQ(&ptrPQ0, &ptrPQ1, &ptrPQ2, fn + ((lowerPart ? lowerPartOffset(offs, j) : upperPartOffset(offs, j)) - 1) * colsFn, cBase, ai, bi);//' - 1' because dim1Fn == Nx - 1(not Nx + 1)
	}}
}}
", Utils.getTypeName<T>());
			string strDefines = string.Format(defines, L, k, Nx, M, colsPQ, colsFn, ksiIsNull ? "true" : "false");
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

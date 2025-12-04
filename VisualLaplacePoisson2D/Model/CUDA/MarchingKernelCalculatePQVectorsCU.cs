using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelCalculatePQVectorsCU
	{
		public static string createProgramCalculatePQVectors(string functionName, string strTypeName)
		{
			string args = "({0}* __restrict__ pq0, {0}* __restrict__ pq1, {0}* __restrict__ pq2, const {0}* __restrict__ fn, {0} cBase, {0} ai, {0} bi)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName;
			string functions =
@"
typedef int (*FuncRow)(int offs, int j);
typedef void (*Action2Arg)({0}* __restrict__ dst, const {0}* __restrict__ src);

__device__ int lowerPartOffset(int offs, int j)
{{
    return offs + j;
}}

__device__ int upperPartOffset(int offs, int j)
{{
    return offs + 1 - j;
}}

__device__ void vectorAssignValue({0} *dst, {0} val)
{{
	for (int i = 0; i < M; i++) dst[i] = val;
}}

__device__ void vectorAssignRow({0}* __restrict__ dst, const {0}* __restrict__ src)
{{
	for (int i = 0; i < M; i++) dst[i] = src[i + 1];
}}

__device__ void vectorAssignShortRow({0}* __restrict__ dst, const {0}* __restrict__ src)
{{
	dst[0] = src[1];
	dst[M - 1] = src[M];
}}

__device__ void vectorAddRow({0}* __restrict__ dst, const {0}* __restrict__ src)
{{
	for (int i = 0; i < M; i++) dst[i] += src[i + 1];
}}

__device__ void vectorAddShortRow({0}* __restrict__ dst, const {0}* __restrict__ src)
{{
	dst[0] += src[1];
	dst[M - 1] += src[M];
}}

__device__ void matrixCMultiplyVector({0}* __restrict__ dst, {0}* __restrict__ src, {0} cBase, {0} ai, {0} bi)
{{
	dst[0] = (cBase * src[0] - bi * src[1]);
	for (int i = 1; i < M - 1; i++)
	{{
		dst[i] = (-ai * src[i - 1] + cBase * src[i] - bi * src[i + 1]);
	}}
	dst[M - 1] = (-ai * src[M - 2] + cBase * src[M - 1]);
}}

__device__ void vectorSubtractVector({0}* __restrict__ dst, {0}* __restrict__ src)
{{
	for (int i = 0; i < M; i++) dst[i] -= src[i];
}}

__device__ void initCalculatePQ({0}* __restrict__ pq1, {0}* __restrict__ pq2, const {0}* __restrict__ fn, bool condition)
{{
	vectorAssignValue(pq1, Zero);
	Action2Arg act = (ksiIsNull && condition) ? vectorAssignShortRow : vectorAssignRow;
	act(pq2, fn);
}}

__device__ void calculatePQ({0}* __restrict__ *pq0, {0}* __restrict__ *pq1, {0}* __restrict__ *pq2, const {0}* __restrict__ fn, {0} cBase, {0} ai, {0} bi)
{{
	{0}* __restrict__ tmp = *pq0;
	*pq0 = *pq1;
	*pq1 = *pq2;
	*pq2 = tmp;

	matrixCMultiplyVector(*pq2, *pq1, cBase, ai, bi);
	vectorSubtractVector(*pq2, *pq0);
	Action2Arg actAddRow = ksiIsNull ? vectorAddShortRow : vectorAddRow;
	actAddRow(*pq2, fn);
}}";

			string programSource =
@"
{{
	int l = blockDim.x * blockIdx.x + threadIdx.x;

	if (l < upperL)
	{{
		bool lowerPart = (l < L);
		int lForOffs = (l >= L) ? l - L : l;
		int offs = lowerPart ? 2 * lForOffs * k : 2 * (lForOffs + 1) * k;
		bool cond = lowerPart ? offs > 1 : offs < Nx - 1;
		int offsFn = lowerPart ? offs + 1 : offs;
		FuncRow rowFn = lowerPart ? lowerPartOffset : upperPartOffset;

		{0} *ptrPQ0 = pq0 + l * colsPQ;
		{0} *ptrPQ1 = pq1 + l * colsPQ;
		{0} *ptrPQ2 = pq2 + l * colsPQ;

		initCalculatePQ(ptrPQ1, ptrPQ2, fn + (offsFn - 1) * colsFn, cond);//' - 1' because dim1Fn == Nx - 1(not Nx + 1)

		for (int j = 2; j <= k; j++) calculatePQ(&ptrPQ0, &ptrPQ1, &ptrPQ2, fn + (rowFn(offs, j) - 1) * colsFn, cBase, ai, bi);//' - 1' because dim1Fn == Nx - 1(not Nx + 1)
	}}
}}";
			return string.Format(functions + strProgramHeader + args + programSource, strTypeName);
		}
	}
}

using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelCalculateYCU
	{
		public static string createProgramCalculateY(string functionName, string strTypeName)
		{
			string args = "({0}* __restrict__ un, const {0}* __restrict__ fn, {0} cBase, {0} ai, {0} bi)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string functions =
@"
typedef void (*Action3Arg)({0}* __restrict__ dst, int rowDst, const {0}* __restrict__ src);

__device__ void matrixCMultiplyArrayVector({0} *_dst, int rowDst, int rowSrc, {0} cBase, {0} ai, {0} bi)
{{
	{0} *dst = _dst + rowDst * colsUn;
	{0} *src = _dst + rowSrc * colsUn;
	dst[1] = cBase * src[1] - bi * src[2];
	for (int i = 2; i < M; i++)
	{{
		dst[i] = -ai * src[i - 1] + cBase * src[i] - bi * src[i + 1];
	}}
	dst[M] = -ai * src[M - 1] + cBase * src[M];
}}

__device__ void arrayRowSubtractRow({0} *_dst, int rowDst, int rowSrc)
{{
	{0} *dst = _dst + rowDst * colsUn;
	{0} *src = _dst + rowSrc * colsUn;
	for (int j = 1; j <= M; j++) dst[j] -= src[j];
}}

__device__ void arrayRowSubtractVector({0}* __restrict__ _dst, int rowDst, const {0}* __restrict__ src)
{{
	{0} *dst = _dst + rowDst * colsUn;
	for (int j = 1; j <= M; j++) dst[j] -= src[j];
}}

__device__ void arrayRowSubtractShortVector({0}* __restrict__ _dst, int rowDst, const {0}* __restrict__ src)
{{
	{0} *dst = _dst + rowDst * colsUn;
	dst[1] -= src[1];
	dst[M] -= src[M];
}}

__device__ void calculateYFromLeftOrRight(int l, {0}* __restrict__ un, const {0}* __restrict__ fn, {0} cBase, {0} ai, {0} bi)//SNE p.232(30)
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
		Action3Arg subtractVector = condSubtraction2 ? arrayRowSubtractShortVector : arrayRowSubtractVector;
		subtractVector(un, i, fn + (i1 - 1) * colsFn);//'- 1' because dim1Fn == Nx - 1(not Nx + 1)
	}}
}}";
			string programSource =
@"
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;

	if (l < upperL)
	{
		calculateYFromLeftOrRight(l, un, fn, cBase, ai, bi);
	}
}";
			return string.Format(functions + strProgramHeader, strTypeName) + programSource;
		}
	}
}


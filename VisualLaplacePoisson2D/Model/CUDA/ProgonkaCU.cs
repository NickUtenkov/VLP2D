using ManagedCuda;

namespace VLP2D.Model
{
	static class ProgonkaCU
	{
		static public string strDefinesProgonkaXY =
@"
static __device__ __constant__ int dimX, dimY, upperX, upperY;
";

		static public string programSourceProgonkaX =
@"
{{
	int j = blockDim.x * blockIdx.x + threadIdx.x + 1;

	if (j <= upperY)
	{{
		for (int i = 1,i1 = dimY; i < dimX - 1; i++,i1 += dimY) unDst[i1 + j] = alphaX[i] * ({0} + unDst[i1 - dimY + j]);//unSrc is used inside strRightSideX
		for (int i = dimX - 2,i1 = dimY * (dimX - 2); i > 0; i--,i1 -= dimY) unDst[i1 + j] += alphaX[i] * unDst[i1 + dimY + j];
	}}
}}
";

		static public string programSourceProgonkaY =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

	if (i <= upperX)
	{{
		i *= dimY;
		for (int j = 1; j < dimY - 1; j++) unDst[i + j] = alphaY[j] * ({0} + unDst[i + j - 1]);//unSrc is used inside strRightSideY
		for (int j = dimY - 2; j > 0; j--) unDst[i + j] += alphaY[j] * unDst[i + j + 1];
	}}
}}
";
	}
}

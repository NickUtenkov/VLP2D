using System;
using VLP2D.Common;

namespace VLP2D.Model.CUDA
{
	internal class EvolutionalFactorizationProgramsCU
	{
		static public string createProgramProgonkaX<T>(string functionName, bool withFn)
		{
			string kernelHeader = UtilsCU.kernelPrefix + functionName;
			string args0 = "({0} *src, {0} *dst, {0} *alphaX, {0} twoDivTau";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = kernelHeader + args;

			string operatorLxx = "(src[(i1 - dimY) + j] - 2.0 * src[i1 + j] + src[(i1 + dimY) + j])";
			string operatorLyy = "(src[i1 + (j - 1)] - 2.0 * src[i1 + j] + src[i1 + (j + 1)])";
			string strRightSideX = string.Format("twoDivTau * ({0} + {1} * ratio + {2})", operatorLxx, operatorLyy, withFn ? "fn[i1 + j] * stepX2" : "0");
			return strProgramHeader + String.Format(programSourceProgonkaX, strRightSideX);
		}

		static public string createProgramProgonkaY<T>(string functionName)
		{
			string kernelHeader = UtilsCU.kernelPrefix + functionName;
			string args0 = "({0} *src, {0} *dst, {0} *srcX, {0} *alphaY, {0} twoDivTau, {0} tau)";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = kernelHeader + args;

			string strRightSideY = "src[i + j] * twoDivTau * hy2";
			return strProgramHeader + String.Format(programSourceProgonkaY, strRightSideY);
		}

		static public string strDefinesProgonkaXY =
@"
static __device__ __constant__ int dimX, dimY, upperX, upperY;
";
		static string programSourceProgonkaX =
@"
{{
	int j = blockDim.x * blockIdx.x + threadIdx.x + 1;

	if (j <= upperY)
	{{
		for (int i = 1,i1 = dimY; i < dimX - 1; i++,i1 += dimY) dst[i1 + j] = alphaX[i] * ({0} + (i != 1 ? dst[i1 - dimY + j] : Zero));//src is used inside strRightSideX
		for (int i = dimX - 2,i1 = dimY * (dimX - 2); i > 0; i--,i1 -= dimY) dst[i1 + j] += alphaX[i] * (i != dimX - 2 ? dst[i1 + dimY + j] : Zero);
	}}
}}
";
		static string programSourceProgonkaY =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;

	if (i <= upperX)
	{{
		i *= dimY;
		for (int j = 1; j < dimY - 1; j++) dst[i + j] = alphaY[j] * ({0} + (j != 1 ? dst[i + j - 1] : Zero));//src is used inside strRightSideY
		for (int j = dimY - 2; j > 0; j--) dst[i + j] += alphaY[j] * (j != dimY - 2 ? dst[i + j + 1] : Zero);
		//post proccess progonka
		for (int j = 1; j <= dimY - 2; j++) dst[i + j] = (srcX[i + j] + tau * dst[i + j]);
	}}
}}
";
	}
}

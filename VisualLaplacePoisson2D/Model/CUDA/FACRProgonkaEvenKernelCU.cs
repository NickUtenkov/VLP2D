using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class FACRProgonkaEvenKernelCU
	{
		public static (CudaKernel, CudaKernel) createKernelProgonkaEven(CudaContext ctx, string typeName, bool bMeeting)
		{
			CUmodule? module;
			string functionNameProgonkaEven = "progonkaEven";
			string functionNameAlfaLengths = "alfaLengths";
			string moduleName = UtilsCU.moduleName(bMeeting ? "FACRProgonkaEvenMeet_" : "FACRProgonkaEvenLeft_", typeName, ctx.DeviceId);

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string definesProgonka =
@"
#define ind(i) (((i) < alfaCount) ? (i) : alfaCount - 1)

static __device__ __constant__ int U, midX, dimY, paramL, countDiagElems, allProgonkaWorkSize, _2tm1;
static __device__ __constant__ {0} stepX2, x2DivY2, x2DivY2Doubled, piDivN2, ck2base;
static __device__ __constant__ float _x2DivY2Doubled, _piDivN2, _ck2base, _Pi4;

__device__ int alfaConvergentCount(float diagElem)
{{
	float u = (diagElem + sqrt(diagElem * diagElem - 4)) / 2;
	float K = ((float)_2tm1) / log2(u);

	return min((midX > 0 ? midX : U), (int)K);//midX can be 0 for not meeting progonka
}}

__device__ void calcAlfa({0} diagElem, {0} *alfa, int alfaCount)
{{
	alfa[0] = One / diagElem;//[SNR] p.75(7)
	for (int i = 1; i < alfaCount; i++) alfa[i] = One / (diagElem - alfa[i - 1]);//[SNR] p.75(7)
}}
";

				string srcProgonkaLeft = 
@"
__device__ void progonka({0} *un, int j, {0} *alfa, {0} coef, int alfaCount)
{{
	//calc beta, put to un
	un[0 * dimY + j] = coef * un[j] * alfa[0];//[SNR] p.75(7)
	for (int i = 1,i1 = i * dimY; i <= U; i++,i1 += dimY) un[i1 + j] = ((coef * un[i1 + j] + un[i1 - dimY + j]) * alfa[ind(i)]);//[SNR] p.75(7)

	for (int i = U - 1,i1 = i * dimY; i >= 0; i--,i1 -= dimY) un[i1 + j] = (un[i1 + j] + (alfa[ind(i)] * un[i1 + dimY + j]));//[SNR] p.75(7)
}}
";
				string srcProgonkaMeet =
@"
__device__ void progonka({0} *un, int j, {0} *alfa, {0} coef, int alfaCount)
{{
	//calc beta, put to un
	for (int i = 0; i <= midX - 1; i++)
	{{
		un[(i + 0) * dimY + j] = (coef * un[(i + 0) * dimY + j] + (i != 0 ? un[(i - 1) * dimY + j] : Zero)) * alfa[ind(i)];
		un[(U - i) * dimY + j] = (coef * un[(U - i) * dimY + j] + (i != 0 ? un[(U - (i - 1)) * dimY + j] : Zero)) * alfa[ind(i)];
	}}

	un[midX * dimY + j] = (un[midX * dimY + j] + alfa[ind(midX - 1)] * un[(midX - 1) * dimY + j]) / (1.0 - alfa[ind(midX - 1)] * alfa[ind(midX - 1)]);//[SNR] p.77(bottom)

	//from middle to left, then from middle to right
	for (int i = midX - 1,i1 = i * dimY; i >= 0; i--,i1 -= dimY) un[i1 + j] += alfa[ind(i + 0)] * un[i1 + dimY + j];
	for (int i = midX + 1,i1 = i * dimY; i <= U; i++,i1 += dimY) un[i1 + j] += alfa[ind(U - i)] * un[i1 - dimY + j];
}}
";
				string argsEven = "({0} *un, {0} *alfaIn, int *alfaOffsets, int *alfaCounts)";
				string srcProgonkaEven =
	@"
{{
	int j0 = blockDim.x * blockIdx.x + threadIdx.x;

	if (j0 >= allProgonkaWorkSize) return;

	{0} *alfa = alfaIn + alfaOffsets[j0];

	{0} PiJDivN2 = (j0 + 1) * piDivN2;
	{0} diagDelta = x2DivY2Doubled * cos(PiJDivN2);
	int j = ((j0 + 1) << paramL) - 1;//even in CPU variant(because of boundaries)

	calcAlfa(ck2base - diagDelta, alfa, alfaCounts[j0]);
	progonka(un, j, alfa, stepX2, alfaCounts[j0]);

	if (paramL >= 1)
	{{
		{0} diagElems[7];//countDiagElems
		diagElems[0] = ck2base + diagDelta;
		if (paramL >= 2)
		{{
			diagElems[1] = -(x2DivY2Doubled * sin(PiJDivN2));
			diagElems[2] = ck2base - diagElems[1];
			diagElems[1] += ck2base;
		}}
		if (paramL >= 3)
		{{
			{0} arg = PiJDivN2 - Pi4;
			diagElems[3] = -(x2DivY2Doubled * cos(arg));//== - diagDelta
			diagElems[4] = ck2base - diagElems[3];//== ck2base + diagDelta
			diagElems[3] += ck2base;//== ck2base - diagDelta

			diagElems[5] = -(x2DivY2Doubled * sin(arg));//== - diagDelta
			diagElems[6] = ck2base - diagElems[5];//== ck2base + diagDelta
			diagElems[5] += ck2base;//== ck2base - diagDelta
		}}

		for (int k = 0; k < countDiagElems; k++)
		{{
			int alfaCount = alfaConvergentCount(to_float(diagElems[k]));
			calcAlfa(diagElems[k], alfa, alfaCount);
			progonka(un, j, alfa, x2DivY2, alfaCount);
		}}
	}}
}}
";
				string argsAlfaLengths = "(int *alfaCounts)";
				string srcAlfaLengths =
@"
{{
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (j >= allProgonkaWorkSize) return;

	float PiJDivN2 = (j + 1) * _piDivN2;
	float minDiag = _ck2base - _x2DivY2Doubled * cos(PiJDivN2);

	if (paramL >= 2)
	{{
		minDiag = min(minDiag, _ck2base - _x2DivY2Doubled * sin(PiJDivN2));
	}}
	if (paramL >= 3)
	{{
		float arg = PiJDivN2 - _Pi4;
		float diagDelta = _x2DivY2Doubled * cos(arg);
		minDiag = min(minDiag, _ck2base - diagDelta);
		minDiag = min(minDiag, _ck2base + diagDelta);

		diagDelta = _x2DivY2Doubled * sin(arg);
		minDiag = min(minDiag, _ck2base - diagDelta);
		minDiag = min(minDiag, _ck2base + diagDelta);
	}}

	alfaCounts[j] = alfaConvergentCount(minDiag);
}}
";
				string strFunctions = definesProgonka + (bMeeting ? srcProgonkaMeet : srcProgonkaLeft);
				string strProgram0 = strFunctions + UtilsCU.kernelPrefix + functionNameProgonkaEven;
				strProgram0 += argsEven + srcProgonkaEven;
				strProgram0 += UtilsCU.kernelPrefix + functionNameAlfaLengths + argsAlfaLengths;
				strProgram0 += srcAlfaLengths;
				string strProgram = string.Format(strProgram0, typeName);
				if (typeName == "float") strProgram = HighPrecisionCU.strSingleDefines + strProgram;
				if (typeName == "double") strProgram = HighPrecisionCU.strDoubleDefines + strProgram;
				if (typeName == "DD128") strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + HighPrecisionCU.strDD128Trig + strProgram;
				if (typeName == "QD256") strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + HighPrecisionCU.strQD256Trig + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}
			return (new CudaKernel(functionNameProgonkaEven, (CUmodule)module) , new CudaKernel(functionNameAlfaLengths, (CUmodule)module));
		}
	}
}

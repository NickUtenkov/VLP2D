using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class VariablesSeparationProgonkaProgramCU
	{
		public static string definesAndConstants()
		{
			string strDefinesAndConstants =
@"
#define RHS(j)			(stepX2 * un[(j)])
#define ind(a) (((a) < alfaCounts[j + offsetJ]) ? (a) : alfaCounts[j + offsetJ] - 1)
static __device__ __constant__ int U, midX;
";
			return strDefinesAndConstants;
		}

		public static string alfaProgram(string functionName, string typeName)
		{
			string strProgramHeader = UtilsCU.kernelPrefix + functionName;
			string args = "({0} *alfaIn, int *alfaOffsets, int *alfaCounts, {0} pi2N2, {0} mult, int dimY)";
			string srcAlfa =
@"
{{
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (j < dimY)
	{{
		{0} *alfa = alfaIn + alfaOffsets[j];

		{0} cosinus = cos((j + 1) * pi2N2);
		{0} diagElem = (One + mult * (One - cosinus)) * 2;
		alfa[0] = 1.0f / diagElem;//[SNR] p.75(7)
		for (int i = 1; i < alfaCounts[j]; i++) alfa[i] = 1.0f / (diagElem - alfa[i - 1]);//[SNR] p.75(7)
	}}
}}
";
			return strProgramHeader + string.Format(args + srcAlfa, typeName);
		}

		public static string meetingProgonkaProgram(string functionName, string typeName)
		{
			string strProgramHeader = UtilsCU.kernelPrefix + functionName;
			string args = "({0} *un, {0} *alfaIn, int *alfaOffsets, int *alfaCounts, {0} stepX2, int offsetJ, int dimY)";
			string srcProgonka =
@"
{{
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (j < dimY)
	{{
		{0} *alfa = alfaIn + alfaOffsets[j + offsetJ];

		//calc beta, put to un
		//from left to middle, than from right to middle
		for (int i = 0; i <= midX - 1; i++)
		{{
			un[(i + 0) * dimY + j] = (RHS((i + 0) * dimY + j) + (i != 0 ? un[(i - 1) * dimY + j] : Zero)) * alfa[ind(i)];
			un[(U - i) * dimY + j] = (RHS((U - i) * dimY + j) + (i != 0 ? un[(U - (i - 1)) * dimY + j] : Zero)) * alfa[ind(i)];
		}}

		un[midX * dimY + j] = (un[(midX) * dimY + j] + alfa[ind(midX - 1)] * un[(midX - 1) * dimY + j]) / (1.0f - alfa[ind(midX - 1)] * alfa[ind(midX - 1)]);

		//from middle to left, than from middle to right
		for (int i = midX - 1,i1 = i * dimY; i >= 0; i--,i1 -= dimY) un[i1 + j] += alfa[ind(i)] * un[i1 + dimY + j];//[SNR] p.75(7)
		for (int i = midX + 1,i1 = i * dimY; i <= U; i++,i1 += dimY) un[i1 + j] += alfa[ind(U - i)] * un[i1 - dimY + j];//[SNR] p.75(7)
	}}
}}
";
			return strProgramHeader + string.Format(args + srcProgonka, typeName);
		}
	}
}

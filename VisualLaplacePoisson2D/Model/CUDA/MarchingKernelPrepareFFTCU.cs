using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelPrepareFFTCU
	{
		public static string createProgramPrepareFFT(string functionName, string strTypeName)
		{
			string args = string.Format("({0}* __restrict__ pq0, const {0}* __restrict__ pq1, const {0}* __restrict__ pq2)", strTypeName);
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSource =
@"
{{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	int m = blockDim.y * blockIdx.y + threadIdx.y;

	if (l < upperL && m < M)
	{
		int idx = (l >= L) ? l - L : l + L;
		pq0[l * colsPQ + m + 1] = pq1[idx * colsPQ + m] - pq2[l * colsPQ + m];
	}
}}";
			return strProgramHeader + programSource;
		}
	}
}

using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelPlaceWToUnCU
	{
		public static string createProgramPlaceWToUn(string functionName, string strTypeName)
		{
			string args = "({0}* __restrict__ _pq0, {0}* __restrict__ _un)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName;
			string programSource =
@"
{{
	int n = blockDim.x * blockIdx.x + threadIdx.x;

	if (n < upperL)
	{{
		{0}* __restrict__ pq0 = _pq0 + n * colsPQ;
		int iUn = n * k + ((n & 1) != 1 ? 1 : k);
		{0}* __restrict__ un = _un + iUn * colsUn;
		for (int m = 1;m <= M; m++) un[m] = pq0[m];
	}}
}}";
			return strProgramHeader + string.Format(args + programSource, strTypeName);
		}
	}
}

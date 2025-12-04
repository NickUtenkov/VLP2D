using Cloo;
using DD128Numeric;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelPlaceWToUnOCL<T>
	{
		public static KernelOCL createKernelPlaceWToUn(DeviceOCL device, ContextOCL context, int L, int k, int M, int colsUn, int colsPQ)
		{
			string functionName = "placeWToUnDF64";
			string args = string.Format("(const global {0}* __restrict__ _pq0, global {0}* __restrict__ _un)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string defines =
@"
#define L				{0}
#define k				{1}
#define M				{2}
#define colsUn			{3}
#define colsPQ			{4}
#define upperL	(L * 2)

";
			string programSource = string.Format(
@"
{{
	int n = get_global_id(0);

	if (n < upperL)
	{{
		const global {0}* __restrict__ pq0 = _pq0 + n * colsPQ;
		int iUn = n * k + ((n & 1) != 1 ? 1 : k);
		global {0}* __restrict__ un = _un + iUn * colsUn;
		for (int m = 1;m <= M; m++) un[m] = pq0[m];
	}}
}}
", Utils.getTypeName<T>());
			string strDefines = string.Format(defines, L, k, M, colsUn, colsPQ);
			string strProgram = strDefines + strProgramHeader + programSource;
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strTypeDefDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strTypeDefQD256 + strProgram;

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, context, device);
			return program.CreateKernel(functionName);
		}
	}
}

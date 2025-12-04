using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MultiGridKernelsCU
	{
		public static string createProgramResidual<T>(string functionName)
		{
			string args = string.Format("({0} *res, {0} *rhs, {0} *un, {0} stepX2, {0} stepY2, int dimY, int upperX, int upperY)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceResidual =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;

	if (i <= upperX && j <= upperY)
	{
		int idx = i * dimY + j;
		int idxim = idx - dimY;//im is 'i minus 1'
		int idxip = idx + dimY;//ip is 'i plus 1'
		int idxjm = idx - 1;//jm is 'j minus 1'
		int idxjp = idx + 1;//jp is 'j plus 1'

		res[idx] = rhs[idx] - (un[idxim] + un[idxip] - 2 * un[idx]) / stepX2 - (un[idxjm] + un[idxjp] - 2 * un[idx]) / stepY2;
	}
}}";
			return strProgramHeader + programSourceResidual;
		}

		public static string createProgramRestrictResidual<T>(string functionName)
		{
			string args = string.Format("({0} *rhs, {0} *res, int rowSizeCoarseGrid, int rowSizeFineGrid, int upperX, int upperY)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceRestrictResidual =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;

	if (i <= upperX && j <= upperY)
	{
		int idx = i * rowSizeCoarseGrid + j;
		int idx1 = 2 * (i * rowSizeFineGrid + j);

		int idxim = idx1 - rowSizeFineGrid;//im is 'i minus 1'
		int idxip = idx1 + rowSizeFineGrid;//ip is 'i plus 1'
		int idxjm = idx1 - 1;//jm is 'j minus 1'
		int idxjp = idx1 + 1;//jp is 'j plus 1'

		rhs[idx] = (4.0 * res[idx1] + (res[idxip] + res[idxim] + res[idxjp] + res[idxjm])) / 8.0;
	}
}}";
			return strProgramHeader + programSourceRestrictResidual;
		}

		public static string createProgramFillArrayWithEdges<T>(string functionName)
		{
			string args = string.Format("({0} *un, {0} val, int upperX, int upperY)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceFillArrayWithEdges =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < upperX && j < upperY)
	{
		int idx = i * upperY + j;

		un[idx] = val;
	}
}}";
			return strProgramHeader + programSourceFillArrayWithEdges;
		}

		public static string createProgramInterpolate<T>(string functionName)
		{
			string args = string.Format("({0} *uf, {0} *uc, int rowSizeFineGrid, int rowSizeCoarseGrid, int upperX, int upperY)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceInterpolate =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i <= upperX && j <= upperY)
	{
		int idxC1 = i * rowSizeCoarseGrid + j;
		int idxC2 = idxC1 + rowSizeCoarseGrid;

		int idx0 = 2 * (i * rowSizeFineGrid + j);//[i + 0, j]
		int idx1 = idx0 + rowSizeFineGrid;//[i + 1, j]

		uf[idx0 + 0] += uc[idxC1];//uc0[i, j]
		uf[idx1 + 0] += 0.50f * (uc[idxC1] + uc[idxC2]);//0.50 * (uc0[i, j] + uc0[i + 1, j])
		uf[idx0 + 1] += 0.50f * (uc[idxC1] + uc[idxC1 + 1]);//0.50 * (uc0[i, j] + uc0[i, j + 1])
		uf[idx1 + 1] += 0.25f * ((uc[idxC1] + uc[idxC1 + 1]) + uc[idxC2] + uc[idxC2 + 1]);//0.25 * (uc0[i, j] + uc0[i, j + 1] + uc0[i + 1, j] + uc0[i + 1, j + 1])
	}
}}";
			return strProgramHeader + programSourceInterpolate;
		}
	}
}

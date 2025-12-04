using VLP2D.Common;

namespace VLP2D.Model
{
	internal class SlidingIterationMultiGridKernelsCU
	{
		readonly string programSource =
@"
{{
	int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2 + params.shiftI + 1;
	int j = (blockDim.y * blockIdx.y + threadIdx.y) * 2 + params.shiftJ + 1;

	if (i <= params.upperX && j <= params.upperY)
	{{
		int idx = i * params.dimY + j;
		int idxim = idx - params.dimY;//im is 'i minus 1'
		int idxip = idx + params.dimY;//ip is 'i plus 1'
		int idxjm = idx - 1;//jm is 'j minus 1'
		int idxjp = idx + 1;//jp is 'j plus 1'

		un[idx] = {0};//cross scheme
	}}
}}";

		public SlidingIterationMultiGridKernelsCU()
		{
		}

		public string createLaplaceEqualStepsProgram(string functionName, string strTypeNameT)//un[i, j] = 0.25 * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1])
		{
			string strAction = "0.25 * (un[idxim] + un[idxip] + un[idxjm] + un[idxjp])";
			string args = string.Format("({0} *un, ExtraParams params)", strTypeNameT);
			return UtilsCU.kernelPrefix + functionName + args + string.Format(programSource, strAction);
		}

		public string createPoissonEqualStepsProgram(string functionName, string strTypeNameT)//un[i, j] = 0.25 * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1] - fn[i, j] * stepX2)
		{
			string strAction = "0.25 * (un[idxim] + un[idxip] + un[idxjm] + un[idxjp] - fn[idx] * stepX2)";
			string args = string.Format("({0} *un, {0} *fn, {0} stepX2, ExtraParams params)", strTypeNameT);
			return UtilsCU.kernelPrefix + functionName + args + string.Format(programSource, strAction);
		}

		public string createLaplaceProgram(string functionName, string strTypeNameT)//un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2)
		{
			string strAction = "coef * ((un[idxim] + un[idxip]) / stepX2 + (un[idxjm] + un[idxjp]) / stepY2)";
			string args = string.Format("({0} *un, {0} coef, {0} stepX2, {0} stepY2, ExtraParams params)", strTypeNameT);
			return UtilsCU.kernelPrefix + functionName + args + string.Format(programSource, strAction);
		}

		public string createPoissonProgram(string functionName, string strTypeNameT)//un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2 - fn[i, j])
		{
			string strAction = "coef * ((un[idxim] + un[idxip]) / stepX2 + (un[idxjm] + un[idxjp]) / stepY2 - fn[idx])";
			string args = string.Format("({0} *un, {0} *fn, {0} coef, {0} stepX2, {0} stepY2, ExtraParams params)", strTypeNameT);
			return UtilsCU.kernelPrefix + functionName + args + string.Format(programSource, strAction);
		}
	}
}

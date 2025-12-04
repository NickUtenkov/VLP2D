using ManagedCuda;

namespace VLP2D.Model
{
	internal class RelaxationKernelsCU : SimpleIterationKernelBaseCU
	{
		public RelaxationKernelsCU(CudaContext ctx, string strTypeName, int sharedDimX, int sharedDimY) : base(ctx, strTypeName, sharedDimX, sharedDimY)
		{
			programSource =
@"
{{
	int ish = threadIdx.x + 1;
	int jsh = threadIdx.y * 2 + 1;
	int id0 = blockDim.x * blockIdx.x + ish;
	int id1 = blockDim.y * blockIdx.y * 2 + jsh;
	__shared__ {1} s[sharedDimX * sharedDimY];

	if (id0 <= upperX && id1 <= upperY)
	{{
		bool odd = (id0 & 1) == 1;
		int deltaIdx = ((val1 == 0) ? (odd ? 0 : 1) : (odd ? 1 : 0));
		int idx = id0 * dimY + id1 + deltaIdx;
		int ind = ish * sharedDimY + jsh + deltaIdx;
		s[ind] = un[idx];

		int idxim = ind - sharedDimY;//im is 'i minus 1'
		int idxip = ind + sharedDimY;//ip is 'i plus 1'
		int idxjm = ind - 1;//jm is 'j minus 1'
		int idxjp = ind + 1;//jp is 'j plus 1'

		s[idxjp] = un[idx + 1];//read value at right
		if (ish == blockDim.x || ish == lastBlockSizeX) s[idxip] = un[idx + dimY];//read below value
		if (ish == 1) s[idxim] = un[idx - dimY];//read above value
		if (jsh == 1) s[idxjm] = un[idx - 1];//read value at left
		__syncthreads();

		un[idx] = {0};//cross scheme
		if (flag[0] != 1 && fabs(un[idx] - s[ind]) > eps) flag[0] = 1;
	}}
}}";
		}

		public CudaKernel createLaplaceEqualStepsSeidelKernel()//un[i, j] = 0.25 * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1])
		{
			string functionName = "Relaxation_LaplaceEqualStepsSeidel";
			string args = "(int val1, {0} *un, int *flag)";
			string strAction = "0.25 * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp])";

			return createKrnl(functionName, args, strAction, "eps");
		}

		public CudaKernel createPoissonEqualStepsSeidelKernel()//un[i, j] = 0.25 * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1] + fn[i, j])
		{
			string functionName = "Relaxation_PoissonEqualStepsSeidel";
			string args = "(int val1, {0} *un, {0} *fn, int *flag)";
			string strAction = "0.25 * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp] + fn[idx])";//fn[idx] is premultiplied by stepX2

			return createKrnl(functionName, args, strAction, "eps");
		}

		public CudaKernel createLaplaceSeidelKernel()//un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2)
		{
			string functionName = "Relaxation_LaplaceSeidel";
			string args = "(int val1, {0} *un, int *flag)";
			string strAction = "coef * ((s[idxim] + s[idxip]) / stepX2 + (s[idxjm] + s[idxjp]) / stepY2)";

			return createKrnl(functionName, args, strAction, "coef, stepX2, stepY2, eps");
		}

		public CudaKernel createPoissonSeidelKernel()//un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2 + fn[i, j])//Pakulina direchlet_num.pdf,p.9(14)
		{
			string functionName = "Relaxation_PoissonSeidel";
			string args = "(int val1, {0} *un, {0} *fn, int *flag)";
			string strAction = "coef * ((s[idxim] + s[idxip]) / stepX2 + (s[idxjm] + s[idxjp]) / stepY2 + fn[idx])";

			return createKrnl(functionName, args, strAction, "coef, stepX2, stepY2, eps");
		}

		public CudaKernel createLaplaceEqualStepsKernel()//un[i, j] = omegaCoef * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1]) + oneMinusOmega * un[i, j];//fn null
		{
			string functionName = "Relaxation_LaplaceEqualSteps";
			string args = "(int val1, {0} *un, int *flag)";
			string strAction = "omegaCoef * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp]) + oneMinusOmega * s[ind]";

			return createKrnl(functionName, args, strAction, "omegaCoef, oneMinusOmega, eps");
		}

		public CudaKernel createPoissonEqualStepsKernel()//un[i, j] = omegaCoef * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1] + fn[i, j]) + oneMinusOmega * un[i, j];//fn already premultiplied by step^2 
		{
			string functionName = "Relaxation_PoissonEqualSteps";
			string args = "(int val1, {0} *un, {0} *fn, int *flag)";
			string strAction = "omegaCoef * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp] + fn[idx]) + oneMinusOmega * s[ind]";//fn[idx] is premultiplied by stepX2

			return createKrnl(functionName, args, strAction, "omegaCoef, oneMinusOmega, eps");
		}

		public CudaKernel createLaplaceKernel()//un[i, j] = omegaCoef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2) + oneMinusOmega * un[i, j];//fn null
		{
			string functionName = "Relaxation_Laplace";
			string args = "(int val1, {0} *un, int *flag)";
			string strAction = "omegaCoef * ((s[idxim] + s[idxip]) / stepX2 + (s[idxjm] + s[idxjp]) / stepY2) + oneMinusOmega * s[ind]";

			return createKrnl(functionName, args, strAction, "omegaCoef, oneMinusOmega, stepX2, stepY2, eps");
		}

		public CudaKernel createPoissonKernel()//un[i, j] = omegaCoef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2 + fn[i, j]) + oneMinusOmega * un[i, j];
		{
			string functionName = "Relaxation_Poisson";
			string args = "(int val1, {0} *un, {0} *fn, int *flag)";
			string strAction = "omegaCoef * ((s[idxim] + s[idxip]) / stepX2 + (s[idxjm] + s[idxjp]) / stepY2 + fn[idx]) + oneMinusOmega * s[ind]";

			return createKrnl(functionName, args, strAction, "omegaCoef, oneMinusOmega, stepX2, stepY2, eps");
		}

		protected override string formatSource(string strAction)
		{
			return string.Format(programSource, strAction, strTypeName);
		}
	}
}

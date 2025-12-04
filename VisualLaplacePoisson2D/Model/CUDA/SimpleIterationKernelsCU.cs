using ManagedCuda;

namespace VLP2D.Model
{
	class SimpleIterationKernelsCU : SimpleIterationKernelBaseCU
	{
		public SimpleIterationKernelsCU(CudaContext ctx, string strTypeName, int sharedDimX, int sharedDimY) : base(ctx, strTypeName, sharedDimX, sharedDimY)
		{
			programSource =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int ish = threadIdx.x + 1;
	int jsh = threadIdx.y + 1;
	__shared__ {1} s[sharedDimX * sharedDimY];

	if (i <= upperX && j <= upperY)
	{{
		int idxO = i * dimY + j;
		int idx = ish * sharedDimY + jsh;
		s[idx] = un0[idxO];

		int idxim = idx - sharedDimY;//im is 'i minus 1'
		int idxip = idx + sharedDimY;//ip is 'i plus 1'
		int idxjm = idx - 1;//jm is 'j minus 1'
		int idxjp = idx + 1;//jp is 'j plus 1'

		if (ish == 1) s[idxim] = un0[idxO - dimY];
		if (ish == blockDim.x || ish == lastBlockSizeX) s[idxip] = un0[idxO + dimY];
		if (jsh == 1) s[idxjm] = un0[idxO - 1];
		if (jsh == blockDim.y || jsh == lastBlockSizeY) s[idxjp] = un0[idxO + 1];
		__syncthreads();

		un1[idxO] = {0};//cross scheme
	}}
}}";
		}

		public CudaKernel createLaplaceEqualStepsKernel()//un1[i, j] = un0[i, j] + tau1 * (Utils.operatorLaplace(un0, i, j) / stepX2)
		{
			string functionName = "SimpleIteration_LaplaceEqualSteps";
			string args = "({0} *un0, {0} *un1, {0} tau)";
			string strAction = "s[idx] + tau * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp] - 4.0 * s[idx]) / stepX2";

			return createKrnl(functionName, args, strAction, "stepX2");
		}

		public CudaKernel createLaplaceEqualStepsNoChebKernel()//un1[i, j] = un0[i, j] + 0.25 * (Utils.operatorLaplace(un0, i, j));//fn == null
		{
			string functionName = "SimpleIteration_LaplaceEqualStepsNoCheb";
			string args = "({0} *un0, {0} *un1)";
			string strAction = "0.25 * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp])";

			return createKrnl(functionName, args, strAction);
		}

		public CudaKernel createLaplaceKernel()//un1[i, j] = un0[i, j] + tau1 * ((un0[i - 1, j] - 2 * un0[i, j] + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] - 2 * un0[i, j] + un0[i, j + 1]) / stepY2);//fn == null
		{
			string functionName = "SimpleIteration_Laplace";
			string args = "({0} *un0, {0} *un1, {0} tau)";
			string strAction = "s[idx] + tau * ((s[idxim] - 2 * s[idx] + s[idxip]) / stepX2 + (s[idxjm] - 2 * s[idx] + s[idxjp]) / stepY2)";

			return createKrnl(functionName, args, strAction, "stepX2, stepY2");
		}

		public CudaKernel createPoissonEqualStepsKernel()//un1[i, j] = un0[i, j] + tau1 * (Utils.operatorLaplace(un0, i, j) / stepX2 + fn[i, j]);//fn is NOT multiplied by step2 
		{
			string functionName = "SimpleIteration_PoissonEqualSteps";
			string args = "({0} *un0, {0} *un1, {0} tau, {0} *fn)";
			string strAction = "s[idx] + tau * ((s[idxim] + s[idxip] + s[idxjm] + s[idxjp] - 4.0 * s[idx]) / stepX2 + fn[idxO])";

			return createKrnl(functionName, args, strAction, "stepX2");
		}

		public CudaKernel createPoissonEqualStepsNoChebKernel()//un1[i, j] = un0[i, j] + 0.25 * (Utils.operatorLaplace(un0, i, j) + fn[i, j])
		{
			string functionName = "SimpleIteration_PoissonEqualStepsNoCheb";
			string args = "({0} *un0, {0} *un1, {0} *fn)";
			string strAction = "0.25 * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp] + fn[idxO] * stepX2)";

			return createKrnl(functionName, args, strAction, "stepX2");
		}

		public CudaKernel createPoissonKernel()//un1[i, j] = un0[i, j] + tau1 * ((un0[i - 1, j] - 2 * un0[i, j] + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] - 2 * un0[i, j] + un0[i, j + 1]) / stepY2 + fn[i, j]);//fn is NOT multiplied by step2 
		{
			string functionName = "SimpleIteration_Poisson";
			string args = "({0} *un0, {0} *un1, {0} tau, {0} *fn)";
			string strAction = "s[idx] + tau * ((s[idxim] - 2 * s[idx] + s[idxip]) / stepX2 + (s[idxjm] - 2 * s[idx] + s[idxjp]) / stepY2 + fn[idxO])";

			return createKrnl(functionName, args, strAction, "stepX2, stepY2");
		}

		protected override string formatSource(string strAction)
		{
			return string.Format(programSource, strAction, strTypeName);
		}
	}
}

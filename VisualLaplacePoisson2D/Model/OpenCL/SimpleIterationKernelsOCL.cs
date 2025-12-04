using Cloo;

namespace VLP2D.Model
{
	internal class SimpleIterationKernelsOCL : SimpleIterationAndRelaxationKernelBaseOCL
	{
		public SimpleIterationKernelsOCL(CommandQueueOCL commands, string strTypeName, int dimX, int dimY, int localDimX, int localDimY, int lastBlockSizeX, int lastBlockSizeY) : base(commands, strTypeName, dimX, dimY, localDimX, localDimY, lastBlockSizeX, lastBlockSizeY)
		{
			programSource =
	@"
{{
	int iLoc = get_local_id(0) + get_global_offset(0);
	int jLoc = get_local_id(1) + get_global_offset(1);
	int i = get_local_size(0) * get_group_id(0) + iLoc;
	int j = get_local_size(1) * get_group_id(1) + jLoc;

	__local {1} s[localDimX * localDimY];

	if (i < dimX - 1 && j < dimY - 1)
	{{
		int idx = i * dimY + j;
		int ind = iLoc * localDimY + jLoc;
		s[ind] = un0[idx];

		int idxim = ind - localDimY;//im is 'i minus 1'
		int idxip = ind + localDimY;//ip is 'i plus 1'
		int idxjm = ind - 1;//jm is 'j minus 1'
		int idxjp = ind + 1;//jp is 'j plus 1'

		if (iLoc == get_global_offset(0)) s[idxim] = un0[idx - dimY];
		if (iLoc == get_local_size(0) || iLoc == lastBlockSizeX) s[idxip] = un0[idx + dimY];
		if (jLoc == get_global_offset(1)) s[idxjm] = un0[idx - 1];
		if (jLoc == get_local_size(1) || jLoc == lastBlockSizeY) s[idxjp] = un0[idx + 1];
		barrier(CLK_LOCAL_MEM_FENCE);

		un1[idx] = {0};//cross scheme
	}}
}}";
		}

		public KernelOCL createLaplaceEqualStepsNoChebKernel()//un1[i, j] = un0[i, j] + 0.25 * (Utils.operatorLaplace(un0, i, j));//fn == null
		{
			string functionName = "SimpleIteration_LaplaceEqualStepsNoCheb";
			string args = "(global {0} *un0, global {0} *un1)";
			string strAction = "0.25 * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp])";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createPoissonEqualStepsNoChebKernel()//un1[i, j] = un0[i, j] + 0.25 * (Utils.operatorLaplace(un0, i, j) + fn[i, j])
		{
			string functionName = "SimpleIteration_PoissonEqualStepsNoCheb";
			string args = "(global {0} *un0, global {0} *un1, global {0} *fn, {0} stepX2)";
			string strAction = "0.25 * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp] + fn[idx] * stepX2)";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createLaplaceEqualStepsKernel()//un1[i, j] = un0[i, j] + tau1 * (Utils.operatorLaplace(un0, i, j) / stepX2)
		{
			string functionName = "SimpleIteration_LaplaceEqualSteps";
			string args = "(global {0} *un0, global {0} *un1, {0} tau, {0} stepX2)";
			string strAction = "s[ind] + tau * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp] - 4 * s[ind]) / stepX2";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createPoissonEqualStepsKernel()//un1[i, j] = un0[i, j] + tau1 * (Utils.operatorLaplace(un0, i, j) / stepX2 + fn[i, j]);//fn is NOT multiplied by step2 
		{
			string functionName = "SimpleIteration_PoissonEqualSteps";
			string args = "(global {0} *un0, global {0} *un1, {0} tau, global {0} *fn, {0} stepX2)";
			string strAction = "s[ind] + tau * ((s[idxim] + s[idxip] + s[idxjm] + s[idxjp] - 4 * s[ind]) / stepX2 + fn[idx])";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createLaplaceKernel()//un1[i, j] = un0[i, j] + tau1 * ((un0[i - 1, j] - 2 * un0[i, j] + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] - 2 * un0[i, j] + un0[i, j + 1]) / stepY2);//fn == null
		{
			string functionName = "SimpleIteration_Laplace";
			string args = "(global {0} *un0, global {0} *un1, {0} tau, {0} stepX2, {0} stepY2)";
			string strAction = "s[ind] + tau * ((s[idxim] - 2 * s[ind] + s[idxip]) / stepX2 + (s[idxjm] - 2 * s[ind] + s[idxjp]) / stepY2)";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createPoissonKernel()//un1[i, j] = un0[i, j] + tau1 * ((un0[i - 1, j] - 2 * un0[i, j] + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] - 2 * un0[i, j] + un0[i, j + 1]) / stepY2 + fn[i, j]);//fn is NOT multiplied by step2 
		{
			string functionName = "SimpleIteration_Poisson";
			string args = "(global {0} *un0, global {0} *un1, {0} tau, global {0} *fn, {0} stepX2, {0} stepY2)";
			string strAction = "s[ind] + tau * ((s[idxim] - 2 * s[ind] + s[idxip]) / stepX2 + (s[idxjm] - 2 * s[ind] + s[idxjp]) / stepY2 + fn[idx])";

			return createKernel(functionName, args, strAction);
		}

		protected override string formatSource(string strAction)
		{
			return string.Format(programSource, strAction, strTypeName);
		}
	}
}

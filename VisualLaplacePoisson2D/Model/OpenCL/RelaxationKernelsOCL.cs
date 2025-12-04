using Cloo;

namespace VLP2D.Model
{
	internal class RelaxationKernelsOCL : SimpleIterationAndRelaxationKernelBaseOCL
	{
		string strCondition;
		public RelaxationKernelsOCL(CommandQueueOCL commands, string strTypeName, int dimX, int dimY, int localDimX, int localDimY, int lastBlockSizeX, int lastBlockSizeY, string strCondition) : base(commands, strTypeName, dimX, dimY, localDimX, localDimY, lastBlockSizeX, lastBlockSizeY)
		{
			programSource =
	@"
{{
	int iLoc = get_local_id(0) + get_global_offset(0);
	int jLoc = get_local_id(1) * 2 + get_global_offset(1);
	int i = get_local_size(0) * get_group_id(0) + iLoc;
	int j = get_local_size(1) * get_group_id(1) * 2 + jLoc;

	__local {0} s[localDimX * localDimY];

	if (i < dimX - 1 && j < dimY - 1)
	{{
		bool odd = (i & 1) == 1;
		int deltaIdx = ((val1 == 0) ? (odd ? 0 : 1) : (odd ? 1 : 0));
		int idx = i * dimY + j + deltaIdx;
		int ind = iLoc * localDimY + jLoc + deltaIdx;
		s[ind] = un[idx];

		int idxim = ind - localDimY;//im is 'i minus 1'
		int idxip = ind + localDimY;//ip is 'i plus 1'
		int idxjm = ind - 1;//jm is 'j minus 1'
		int idxjp = ind + 1;//jp is 'j plus 1'

		s[idxjp] = un[idx + 1];//read value at right
		if (iLoc == get_local_size(0) || iLoc == lastBlockSizeX) s[idxip] = un[idx + dimY];//read below value
		if (iLoc == get_global_offset(0)) s[idxim] = un[idx - dimY];//read above value
		if (jLoc == get_global_offset(1)) s[idxjm] = un[idx - 1];//read value at left
		barrier(CLK_LOCAL_MEM_FENCE);

		un[idx] = {1};//cross scheme
		if (flag[0] != 1 && {2}) flag[0] = 1;
	}}
}}";
			this.strCondition = strCondition;
		}

		public KernelOCL createLaplaceEqualStepsSeidelKernel()//un[i, j] = 0.25 * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1])
		{
			string functionName = "Relaxation_LaplaceEqualStepsSeidel";
			string args = "(int val1, global {0} *un, {0} eps, global int *flag)";
			string strAction = "0.25 * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp])";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createPoissonEqualStepsSeidelKernel()//un[i, j] = 0.25 * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1] + fn[i, j])
		{
			string functionName = "Relaxation_PoissonEqualStepsSeidel";
			string args = "(int val1, global {0} *un, global {0} *fn, {0} eps, global int *flag)";
			string strAction = "0.25 * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp] + fn[idx])";//fn[idx] is premultiplied by stepX2

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createLaplaceSeidelKernel()//un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2)
		{
			string functionName = "Relaxation_LaplaceSeidel";
			string args = "(int val1, global {0} *un, {0} coef, {0} stepX2, {0} stepY2, {0} eps, global int *flag)";
			string strAction = "coef * ((s[idxim] + s[idxip]) / stepX2 + (s[idxjm] + s[idxjp]) / stepY2)";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createPoissonSeidelKernel()//un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2 + fn[i, j])//Pakulina direchlet_num.pdf,p.9(14)
		{
			string functionName = "Relaxation_PoissonSeidel";
			string args = "(int val1, global {0} *un, global {0} *fn, {0} coef, {0} stepX2, {0} stepY2, {0} eps, global int *flag)";
			string strAction = "coef * ((s[idxim] + s[idxip]) / stepX2 + (s[idxjm] + unsidxjp]) / stepY2 + fn[idx])";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createLaplaceEqualStepsKernel()//un[i, j] = omegaCoef * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1]) + oneMinusOmega * un[i, j];//fn null
		{
			string functionName = "Relaxation_LaplaceEqualSteps";
			string args = "(int val1, global {0} *un, {0} omegaCoef, {0} oneMinusOmega, {0} eps, global int *flag)";
			string strAction = "omegaCoef * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp]) + oneMinusOmega * s[ind]";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createPoissonEqualStepsKernel()//un[i, j] = omegaCoef * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1] + fn[i, j]) + oneMinusOmega * un[i, j];//fn already premultiplied by step^2 
		{
			string functionName = "Relaxation_PoissonEqualSteps";
			string args = "(int val1, global {0} *un, {0} omegaCoef, {0} oneMinusOmega, global {0} *fn, {0} eps, global int *flag)";
			string strAction = "omegaCoef * (s[idxim] + s[idxip] + s[idxjm] + s[idxjp] + fn[idx]) + oneMinusOmega * s[ind]";//fn[idx] is premultiplied by stepX2

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createLaplaceKernel()//un[i, j] = omegaCoef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2) + oneMinusOmega * un[i, j];//fn null
		{
			string functionName = "Relaxation_Laplace";
			string args = "(int val1, global {0} *un, {0} omegaCoef, {0} oneMinusOmega, {0} stepX2, {0} stepY2, {0} eps, global int *flag)";
			string strAction = "omegaCoef * ((s[idxim] + s[idxip]) / stepX2 + (s[idxjm] + s[idxjp]) / stepY2) + oneMinusOmega * s[ind]";

			return createKernel(functionName, args, strAction);
		}

		public KernelOCL createPoissonKernel()//un[i, j] = omegaCoef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2 + fn[i, j]) + oneMinusOmega * un[i, j];
		{
			string functionName = "Relaxation_Poisson";
			string args = "(int val1, global {0} *un, {0} omegaCoef, {0} oneMinusOmega, global {0} *fn, {0} stepX2, {0} stepY2, {0} eps, global int *flag)";
			string strAction = "omegaCoef * ((s[idxim] + s[idxip]) / stepX2 + (s[idxjm] + s[idxjp]) / stepY2 + fn[idx]) + oneMinusOmega * s[ind]";

			return createKernel(functionName, args, strAction);
		}

		protected override string formatSource(string strAction)
		{
			return string.Format(programSource, strTypeName, strAction, strCondition);
		}
	}
}

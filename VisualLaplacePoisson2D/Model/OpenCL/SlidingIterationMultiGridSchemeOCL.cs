
using Cloo;
using DD128Numeric;
using QD256Numeric;
using System.Collections.Generic;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class SlidingIterationMultiGridSchemeOCL<T> where T : struct, INumber<T>
	{
		readonly CommandQueueOCL commands;
		readonly int[] order = [ 0, 3, 1, 2 ];
		KernelOCL[] kernels = { null, null, null, null };
		List<KernelParams> listParams = new List<KernelParams>();
		class KernelParams
		{
			public KernelOCL kernel;
			public object[][] args;
			public long[][] workSize;

			public KernelParams()
			{
				args = new object[4][];
				workSize = new long[4][];
			}
		}

		readonly string programSource =
@"
{{
	int i = get_global_id(0) * 2 + shiftI + 1;
	int j = get_global_id(1) * 2 + shiftJ + 1;

	int idx = i * dimY + j;
	int idxim = idx - dimY;//im is 'i minus 1'
	int idxip = idx + dimY;//ip is 'i plus 1'
	int idxjm = idx - 1;//jm is 'j minus 1'
	int idxjp = idx + 1;//jp is 'j plus 1'

	un[idx] = {0};//cross scheme
}}";

		public SlidingIterationMultiGridSchemeOCL(CommandQueueOCL commands)
		{
			this.commands = commands;
		}

		public void doIterations(int level, int nIterations)//O(h^2);sliding iteration method Gauss-Seidel;sequential shift method;Libman method
		{
			KernelOCL kernel = listParams[level].kernel;
			for (int i = 0; i < nIterations; i++)
			{
				foreach (int j in order)
				{
					if (listParams[level].workSize[j][0] == 0 || listParams[level].workSize[j][1] == 0) continue;
					UtilsCL.setKernelArguments<T>(kernel, listParams[level].args[j]);
					commands.Execute(kernel, null, listParams[level].workSize[j], null, null);
				}
			}
		}

		public void addKernelParameters(int dimX, int dimY, BufferOCL<T> un, BufferOCL<T> fn, T stepX, T stepY)
		{
			T stepX2 = stepX * stepX;
			T stepY2 = stepY * stepY;

			KernelParams kernelParams = new KernelParams();
			listParams.Add(kernelParams);

			long wx1 = (dimX - 2) / 2;
			long wx0 = wx1 + (((dimX - 2) % 2 == 0) ? 0 : 1);
			long wy1 = (dimY - 2) / 2;
			long wy0 = wy1 + (((dimY - 2) % 2 == 0) ? 0 : 1);
			kernelParams.workSize[0] = [wx0, wy0];
			kernelParams.workSize[3] = [wx1, wy1];
			kernelParams.workSize[1] = [wx0, wy1];
			kernelParams.workSize[2] = [wx1, wy0];

			string strTypeNameT = Utils.getTypeName<T>();
			bool equalSteps = T.Abs(stepX - stepY) < T.CreateTruncating(1E-15);
			if (equalSteps)
			{
				if (fn == null)
				{
					string functionName = "LaplaceEqualStepsMG";
					if (kernels[0] == null) kernels[0] = createLaplaceEqualStepsKernel(functionName, strTypeNameT);
					kernelParams.kernel = kernels[0];
					kernelParams.args[0] = [un, 0, 0, dimY];
					kernelParams.args[1] = [un, 0, 1, dimY];
					kernelParams.args[2] = [un, 1, 0, dimY];
					kernelParams.args[3] = [un, 1, 1, dimY];
				}
				else
				{
					string functionName = "PoissonEqualStepsMG";
					if (kernels[1] == null) kernels[1] = createPoissonEqualStepsKernel(functionName, strTypeNameT);
					kernelParams.kernel = kernels[1];
					kernelParams.args[0] = [un, fn, stepX2, 0, 0, dimY];
					kernelParams.args[1] = [un, fn, stepX2, 0, 1, dimY];
					kernelParams.args[2] = [un, fn, stepX2, 1, 0, dimY];
					kernelParams.args[3] = [un, fn, stepX2, 1, 1, dimY];
				}
			}
			else
			{
				T _2 = T.CreateTruncating(2.0);
				T coef = T.One / (_2 / stepX2 + _2 / stepY2);
				if (fn == null)
				{
					string functionName = "LaplaceMG";
					if (kernels[2] == null) kernels[2] = createLaplaceKernel(functionName, strTypeNameT);
					kernelParams.kernel = kernels[2];
					kernelParams.args[0] = [un, coef, stepX2, stepY2, 0, 0, dimY];
					kernelParams.args[1] = [un, coef, stepX2, stepY2, 0, 1, dimY];
					kernelParams.args[2] = [un, coef, stepX2, stepY2, 1, 0, dimY];
					kernelParams.args[3] = [un, coef, stepX2, stepY2, 1, 1, dimY];
				}
				else
				{
					string functionName = "PoissonMG";
					if (kernels[3] == null) kernels[3] = createPoissonKernel(functionName, strTypeNameT);
					kernelParams.kernel = kernels[3];
					kernelParams.args[0] = [un, fn, coef, stepX2, stepY2, 0, 0, dimY];
					kernelParams.args[1] = [un, fn, coef, stepX2, stepY2, 0, 1, dimY];
					kernelParams.args[2] = [un, fn, coef, stepX2, stepY2, 1, 0, dimY];
					kernelParams.args[3] = [un, fn, coef, stepX2, stepY2, 1, 1, dimY];
				}
			}
		}

		public void cleanup()
		{
			for (int i = 0; i < kernels.Length; i++) UtilsCL.disposeKP(ref kernels[i], true);
		}

		KernelOCL createLaplaceEqualStepsKernel(string functionName, string strTypeNameT)//un[i, j] = 0.25 * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1])
		{
			string strAction = "0.25 * (un[idxim] + un[idxip] + un[idxjm] + un[idxjp])";
			string args = string.Format("(global {0} *un, int shiftI, int shiftJ, int dimY)", strTypeNameT);
			return createKernel(functionName, args, programSource, strAction);
		}

		KernelOCL createPoissonEqualStepsKernel(string functionName, string strTypeNameT)//un[i, j] = 0.25 * (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1] - fn[i, j] * stepX2)
		{
			string strAction = "0.25 * (un[idxim] + un[idxip] + un[idxjm] + un[idxjp] - fn[idx] * stepX2)";
			string args = string.Format("(global {0} *un, global {0} *fn, {0} stepX2, int shiftI, int shiftJ, int dimY)", strTypeNameT);
			return createKernel(functionName, args, programSource, strAction);
		}

		KernelOCL createLaplaceKernel(string functionName, string strTypeNameT)//un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2)
		{
			string strAction = "coef * ((un[idxim] + un[idxip]) / stepX2 + (un[idxjm] + un[idxjp]) / stepY2)";
			string args = string.Format("(global {0} *un, {0} coef, {0} stepX2, {0} stepY2, int shiftI, int shiftJ, int dimY)", strTypeNameT);
			return createKernel(functionName, args, programSource, strAction);
		}

		KernelOCL createPoissonKernel(string functionName, string strTypeNameT)//un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2 - fn[i, j])
		{
			string strAction = "coef * ((un[idxim] + un[idxip]) / stepX2 + (un[idxjm] + un[idxjp]) / stepY2 - fn[idx])";
			string args = string.Format("(global {0} *un, global {0} *fn, {0} coef, {0} stepX2, {0} stepY2, int shiftI, int shiftJ, int dimY)", strTypeNameT);
			return createKernel(functionName, args, programSource, strAction);
		}

		KernelOCL createKernel(string functionName, string args, string programSource, string strAction)
		{
			if (typeof(T) != typeof(float) && typeof(T) != typeof(double)) strAction = ArithmeticReplacer.replaceArithmeticOperators(strAction);
			string strProgram = UtilsCL.kernelPrefix + functionName + args + string.Format(programSource, strAction);
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			ProgramOCL program = UtilsCL.createProgram(strProgram, null, commands.Context, commands.Device);
			return program.CreateKernel(functionName);
		}
	}
}

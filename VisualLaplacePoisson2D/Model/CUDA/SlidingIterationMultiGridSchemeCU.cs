using DD128Numeric;
using ManagedCuda;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class SlidingIterationMultiGridSchemeCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>
	{
		readonly int[] order = [ 0, 3, 1, 2 ];
		SlidingIterationMultiGridKernelsCU kernelCreator;
		string strProgram = "struct ExtraParams {int shiftI, shiftJ, dimY, upperX, upperY;};";
		CudaKernel[] kernels = { null, null, null, null };
		CudaContext ctx;
		string strTypeNameT;
		List<KernelParams> listParams = new List<KernelParams>();
		class KernelParams
		{
			public CudaKernel kernel;
			public object[][] args;
			public int[][] workSize;

			public KernelParams()
			{
				args = new object[4][];
				workSize = new int[4][];
			}
		}
		struct ExtraParams(int i, int j, int dimY, int upperX, int upperY)
		{
			int i = i, j = j, dimY = dimY, upperX = upperX, upperY = upperY;
		}

		public SlidingIterationMultiGridSchemeCU(CudaContext ctx)
		{
			this.ctx = ctx;
			strTypeNameT = Utils.getTypeName<T>();
			kernelCreator = new SlidingIterationMultiGridKernelsCU();
		}

		public void addKernelParameters(int dimX, int dimY, CudaDeviceVariable<T> un, CudaDeviceVariable<T> fn, T stepX, T stepY)
		{
			T stepX2 = stepX * stepX;
			T stepY2 = stepY * stepY;

			KernelParams kernelParams = new KernelParams();
			listParams.Add(kernelParams);

			int wx1 = (dimX - 2) / 2;
			int wx0 = wx1 + (((dimX - 2) % 2 == 0) ? 0 : 1);
			int wy1 = (dimY - 2) / 2;
			int wy0 = wy1 + (((dimY - 2) % 2 == 0) ? 0 : 1);
			kernelParams.workSize[0] = [wx0, wy0];
			kernelParams.workSize[3] = [wx1, wy1];
			kernelParams.workSize[1] = [wx0, wy1];
			kernelParams.workSize[2] = [wx1, wy0];

			bool equalSteps = T.Abs(stepX - stepY) < T.CreateTruncating(1E-15);
			if (equalSteps)
			{
				if (fn == null)
				{
					string functionName = "LaplaceEqualStepsMG";
					if (kernels[0] == null) kernels[0] = createKernel(kernelCreator.createLaplaceEqualStepsProgram, functionName);
					kernelParams.kernel = kernels[0];
					kernelParams.args[0] = [un.DevicePointer, createExtraParams(0, 0, dimY, kernelParams.workSize)];
					kernelParams.args[1] = [un.DevicePointer, createExtraParams(0, 1, dimY, kernelParams.workSize)];
					kernelParams.args[2] = [un.DevicePointer, createExtraParams(1, 0, dimY, kernelParams.workSize)];
					kernelParams.args[3] = [un.DevicePointer, createExtraParams(1, 1, dimY, kernelParams.workSize)];
				}
				else
				{
					string functionName = "PoissonEqualStepsMG";
					if (kernels[1] == null) kernels[1] = createKernel(kernelCreator.createPoissonEqualStepsProgram, functionName);
					kernelParams.kernel = kernels[1];
					kernelParams.args[0] = [un.DevicePointer, fn.DevicePointer, stepX2, createExtraParams(0, 0, dimY, kernelParams.workSize)];
					kernelParams.args[1] = [un.DevicePointer, fn.DevicePointer, stepX2, createExtraParams(0, 1, dimY, kernelParams.workSize)];
					kernelParams.args[2] = [un.DevicePointer, fn.DevicePointer, stepX2, createExtraParams(1, 0, dimY, kernelParams.workSize)];
					kernelParams.args[3] = [un.DevicePointer, fn.DevicePointer, stepX2, createExtraParams(1, 1, dimY, kernelParams.workSize)];
				}
			}
			else
			{
				T _2 = T.CreateTruncating(2.0);
				T coef = T.One / (_2 / stepX2 + _2 / stepY2);
				if (fn == null)
				{
					string functionName = "LaplaceMG";
					if (kernels[2] == null) kernels[2] = createKernel(kernelCreator.createLaplaceProgram, functionName);
					kernelParams.kernel = kernels[2];
					kernelParams.args[0] = [un.DevicePointer, coef, stepX2, stepY2, createExtraParams(0, 0, dimY, kernelParams.workSize)];
					kernelParams.args[1] = [un.DevicePointer, coef, stepX2, stepY2, createExtraParams(0, 1, dimY, kernelParams.workSize)];
					kernelParams.args[2] = [un.DevicePointer, coef, stepX2, stepY2, createExtraParams(1, 0, dimY, kernelParams.workSize)];
					kernelParams.args[3] = [un.DevicePointer, coef, stepX2, stepY2, createExtraParams(1, 1, dimY, kernelParams.workSize)];
				}
				else
				{
					string functionName = "PoissonMG";
					if (kernels[3] == null) kernels[3] = createKernel(kernelCreator.createPoissonProgram, functionName);
					kernelParams.kernel = kernels[3];
					kernelParams.args[0] = [un.DevicePointer, fn.DevicePointer, coef, stepX2, stepY2, createExtraParams(0, 0, dimY, kernelParams.workSize)];
					kernelParams.args[1] = [un.DevicePointer, fn.DevicePointer, coef, stepX2, stepY2, createExtraParams(0, 1, dimY, kernelParams.workSize)];
					kernelParams.args[2] = [un.DevicePointer, fn.DevicePointer, coef, stepX2, stepY2, createExtraParams(1, 0, dimY, kernelParams.workSize)];
					kernelParams.args[3] = [un.DevicePointer, fn.DevicePointer, coef, stepX2, stepY2, createExtraParams(1, 1, dimY, kernelParams.workSize)];
				}
			}
		}

		public void doIterations(int level, int nIterations)//O(h^2);sliding iteration method Gauss-Seidel;sequential shift method;Libman method
		{
			CudaKernel kernel = listParams[level].kernel;
			for (int i = 0; i < nIterations; i++)
			{
				foreach (int j in order)
				{
					if (listParams[level].workSize[j][0] == 0 || listParams[level].workSize[j][1] == 0) continue;
					UtilsCU.set2DKernelDims(kernel, listParams[level].workSize[j][0], listParams[level].workSize[j][1]);
					kernel.Run(listParams[level].args[j]);
				}
			}
		}

		public void cleanup()
		{
			for (int i = 0; i < kernels.Length; i++) if (kernels[i] != null) ctx?.UnloadKernel(kernels[i]);
			ctx = null;
		}

		ExtraParams createExtraParams(int i, int j, int dimY, int[][] workSize)
		{
			int idx = i * 2 + j;
			return new ExtraParams(i, j, dimY, 2 * workSize[idx][0], 2 * workSize[idx][1]);
		}

		CudaKernel createKernel(Func<string, string, string> func, string functionName)
		{
			string strPrg = strProgram;
			if (typeof(T) == typeof(DD128)) strPrg = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strPrg;
			if (typeof(T) == typeof(QD256)) strPrg = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strPrg;
			return UtilsCU.createKernel(strPrg + func(functionName, strTypeNameT), functionName, ctx);
		}
	}
}

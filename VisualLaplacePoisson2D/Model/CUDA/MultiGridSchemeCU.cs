using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using System;
using System.Diagnostics;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class MultiGridSchemeCU<T> : Iterative1DScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>
	{
		CudaContext ctx;
		readonly CudaDeviceVariable<T>[] un, rhs, res;//right hand side,residual
		CudaDeviceVariable<T> fn;
		readonly T[] stepX, stepY;
		readonly SlidingIterationMultiGridSchemeCU<T> smoother;
		readonly T eps;
		readonly int nLevels;
		readonly int[] dim0, dim1;
		T[] un0;
		int[][] workSize2DInternalPoints, workSize2DAllPoints, workSize1DInternalPointsX, workSize1DInternalPointsY;
		int[] workSizeEps = { 0, 0 }, workOffsetEps = { 0, 0 };
		CudaDeviceVariable<int> flagCU;
		bool un0Changed = false;
		const int countSmoothingIterations = 3;//if 1 then no convergence in rectangle cases

		CudaKernel kernelResidual, kernelRestrictResidual, kernelFillArrayWithEdges, kernelInterpolate, kernelEpsExceeded;
		object[] argsResidual, argsRestrictResidual, argsFillArrayWithEdges, argsInterpolate, argsEpsExceeded;

		public MultiGridSchemeCU(int cXSegments, int cYSegments, T stepXIn, T stepYIn, Func<T, T, T> fKsi, T eps, int cudaDevice)
		{
			ctx = new CudaContext(cudaDevice);
			dimX = cXSegments + 1;
			dimY = cYSegments + 1;
			this.eps = eps;

			un0 = new T[dimX * dimY];

			if (fKsi != null)
			{
				GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => un0[i * dimY + j] = -fKsi(stepXIn * T.CreateTruncating(i), stepYIn * T.CreateTruncating(j)));
				fn = un0;
			}

			int nLevelsX = (int)Math.Log(cXSegments, 2);
			int nLevelsY = (int)Math.Log(cYSegments, 2);
			nLevels = Math.Min(nLevelsX, nLevelsY);

			dim0 = new int[nLevels];
			dim1 = new int[nLevels];
			un = new CudaDeviceVariable<T>[nLevels];
			rhs = new CudaDeviceVariable<T>[nLevels];
			res = new CudaDeviceVariable<T>[nLevels];
			stepX = new T[nLevels];
			stepY = new T[nLevels];
			smoother = new SlidingIterationMultiGridSchemeCU<T>(ctx);
			workSize2DInternalPoints = new int[nLevels][];
			workSize2DAllPoints = new int[nLevels][];
			workSize1DInternalPointsX = new int[nLevels][];
			workSize1DInternalPointsY = new int[nLevels][];

			int cSegsX = cXSegments;//is 2^n
			int cSegsY = cYSegments;//is 2^n
			T lngX = stepXIn * T.CreateTruncating(cSegsX);
			T lngY = stepYIn * T.CreateTruncating(cSegsY);
			try
			{
				for (int i = 0; i < nLevels; i++)
				{
					dim0[i] = cSegsX + 1;
					dim1[i] = cSegsY + 1;
					un[i] = new CudaDeviceVariable<T>(dim0[i] * dim1[i]);
					rhs[i] = (i == 0) && (fn != null) ? fn : new CudaDeviceVariable<T>(dim0[i] * dim1[i]);
					res[i] = (i < nLevels - 1) ? new CudaDeviceVariable<T>(dim0[i] * dim1[i]) : null;
					stepX[i] = lngX / T.CreateTruncating(cSegsX);
					stepY[i] = lngY / T.CreateTruncating(cSegsY);
					smoother.addKernelParameters(dim0[i], dim1[i], un[i], (i == 0) ? fn : rhs[i], stepX[i], stepY[i]);
					cSegsX /= 2;
					cSegsY /= 2;
					workSize2DInternalPoints[i] = [dim0[i] - 2, dim1[i] - 2];
					if (i > 0) workSize2DAllPoints[i] = [dim0[i], dim1[i]];
					if (i > 0) workSize1DInternalPointsX[i] = [dim0[i] - 2];
					if (i > 0) workSize1DInternalPointsY[i] = [dim1[i] - 2];
				}
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			if (fn == null) ctx.ClearMemory(rhs[0].DevicePointer, 0, rhs[0].SizeInBytes);
			flagCU = new CudaDeviceVariable<int>(1);

			CUmodule? module;
			string moduleName = UtilsCU.moduleName("MultiGrid_", Utils.getTypeName<T>(), ctx.DeviceId);
			string functionNameResidual = "Residual";
			string functionNameRestrictResidual = "RestrictResidual";
			string functionNameFillArrayWithEdges = "FillArrayWithEdges";
			string functionNameInterpolate = "Interpolate";
			string functionNameEpsExceeded = "EpsExceeded";

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string strProgram = "";

				strProgram += MultiGridKernelsCU.createProgramResidual<T>(functionNameResidual);
				strProgram += MultiGridKernelsCU.createProgramRestrictResidual<T>(functionNameRestrictResidual);
				strProgram += MultiGridKernelsCU.createProgramFillArrayWithEdges<T>(functionNameFillArrayWithEdges);
				strProgram += MultiGridKernelsCU.createProgramInterpolate<T>(functionNameInterpolate);
				strProgram += UtilsCU.createEpsExceededProgram<T>(dimX, dimY, functionNameEpsExceeded);

				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}

			kernelResidual = new CudaKernel(functionNameResidual, (CUmodule)module);
			kernelRestrictResidual = new CudaKernel(functionNameRestrictResidual, (CUmodule)module);
			kernelFillArrayWithEdges = new CudaKernel(functionNameFillArrayWithEdges, (CUmodule)module);
			kernelInterpolate = new CudaKernel(functionNameInterpolate, (CUmodule)module);
			kernelEpsExceeded = new CudaKernel(functionNameEpsExceeded, (CUmodule)module);
			UtilsCU.set2DKernelDims(kernelEpsExceeded, workSize2DInternalPoints[0][0], workSize2DInternalPoints[0][1]);

			argsResidual = [res[0].DevicePointer, rhs[0].DevicePointer, un[0].DevicePointer, stepX[0] * stepX[0], stepY[0] * stepY[0], dim1[0], 0, 0];
			argsRestrictResidual = [rhs[0].DevicePointer, res[0].DevicePointer, dim1[0 + 1], dim1[0 + 0], 0, 0];
			argsFillArrayWithEdges = [un[0].DevicePointer, 0, 0, 0];
			argsInterpolate = [un[0 + 0].DevicePointer, un[0 + 1].DevicePointer, dim1[0 + 1], dim1[0 + 0], 0, 0];
			argsEpsExceeded = [un[0].DevicePointer, res[0].DevicePointer, flagCU.DevicePointer, 0, eps];
		}

		~MultiGridSchemeCU()
		{
		}

		public T doIteration(int iter)
		{
			VCycle(0);
			un0Changed = true;

			return epsExceeded() ? eps + eps : T.Zero;
		}

		void VCycle(int startLevel)
		{
			for (int level = startLevel; level < nLevels - 1; level++)
			{
				smoother.doIterations(level, countSmoothingIterations);//fn(level=0),rhs[level](level>0),un[level] -> un[level]
				residual(level);//rhs[level],un[level] -> res[level]
				restrictResidual(level);//res[level + 0] -> rhs[level + 1]
				fillArrayWithEdges(level + 1, T.Zero);//un[level + 1] edges to val = 0
			}
			smoother.doIterations(nLevels - 1, countSmoothingIterations);
			for (int level = nLevels - 2; level >= startLevel; level--)
			{
				interpolate(level);//un[level + 1] -> un[level + 0]
				if (level == 0) res[0].CopyToDevice(un[0]);
				smoother.doIterations(level, countSmoothingIterations);
			}
		}

		void residual(int level)
		{
			argsResidual[0] = res[level].DevicePointer;
			argsResidual[1] = rhs[level].DevicePointer;
			argsResidual[2] = un[level].DevicePointer;
			argsResidual[3] = stepX[level] * stepX[level];
			argsResidual[4] = stepY[level] * stepY[level];
			argsResidual[5] = dim1[level];
			argsResidual[6] = workSize2DInternalPoints[level][0];
			argsResidual[7] = workSize2DInternalPoints[level][1];

			UtilsCU.set2DKernelDims(kernelResidual, workSize2DInternalPoints[level][0], workSize2DInternalPoints[level][1]);
			kernelResidual.Run(argsResidual);
		}

		void restrictResidual(int level)//Restriction
		{
			argsRestrictResidual[0] = rhs[level + 1].DevicePointer;
			argsRestrictResidual[1] = res[level + 0].DevicePointer;
			argsRestrictResidual[2] = dim1[level + 1];//coarse
			argsRestrictResidual[3] = dim1[level + 0];//fine
			argsRestrictResidual[4] = workSize2DInternalPoints[level + 1][0];
			argsRestrictResidual[5] = workSize2DInternalPoints[level + 1][1];

			UtilsCU.set2DKernelDims(kernelRestrictResidual, workSize2DInternalPoints[level + 1][0], workSize2DInternalPoints[level + 1][1]);
			kernelRestrictResidual.Run(argsRestrictResidual);
		}

		void fillArrayWithEdges(int level, T val)//https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueFillBuffer.html
		{
			argsFillArrayWithEdges[0] = un[level].DevicePointer;
			argsFillArrayWithEdges[1] = val;
			argsFillArrayWithEdges[2] = workSize2DAllPoints[level][0];
			argsFillArrayWithEdges[3] = workSize2DAllPoints[level][1];

			UtilsCU.set2DKernelDims(kernelFillArrayWithEdges, workSize2DAllPoints[level][0], workSize2DAllPoints[level][1]);
			kernelFillArrayWithEdges.Run(argsFillArrayWithEdges);
		}

		void interpolate(int level)//Prolongation
		{//level + 0 - fine grid, level + 1 - coarse grid
			argsInterpolate[0] = un[level + 0].DevicePointer;//fine grid
			argsInterpolate[1] = un[level + 1].DevicePointer;//coarse grid
			argsInterpolate[2] = dim1[level + 0];//fine grid
			argsInterpolate[3] = dim1[level + 1];//coarse grid
			int upperX = workSize2DInternalPoints[level + 1][0]; int upperY = workSize2DInternalPoints[level + 1][1];
			argsInterpolate[4] = upperX;//coarse grid
			argsInterpolate[5] = upperY;//coarse grid

			UtilsCU.set2DKernelDims(kernelInterpolate, upperX + 1, upperY + 1);
			kernelInterpolate.Run(argsInterpolate);
		}

		bool epsExceeded()
		{
			int cParts = 128;
			flagCU[0] = 0;

			if (workSize2DInternalPoints[0][0] <= cParts)
			{
				kernelEpsExceeded.Run(argsEpsExceeded);
			}
			else
			{
				workOffsetEps[0] = 0;
				workSizeEps[1] = workSize2DInternalPoints[0][1];
				int wSize = (int)workSize2DInternalPoints[0][0];
				int partSize = wSize / cParts;
				workSizeEps[0] = partSize;
				int remainder = wSize % cParts;
				if (remainder > 0) cParts++;
				for (int i = 0; i < cParts; i++)
				{
					UtilsCU.set2DKernelDims(kernelEpsExceeded, workSizeEps[0], workSizeEps[1]);
					argsEpsExceeded[3] = workOffsetEps[0];
					kernelEpsExceeded.Run(argsEpsExceeded);
					if (flagCU[0] == 1) break;

					if ((i == cParts - 1) && (remainder != 0)) workSizeEps[0] = remainder;
					workOffsetEps[0] += partSize;
				}
			}

			return flagCU[0] == 1;
		}

		public override T[] getArray()
		{
			if (un0Changed)
			{
				un0Changed = false;
				un[0].CopyToHost(un0);
			}

			return un0;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			un[0].CopyToDevice(un0);
		}

		public int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			if (kernelResidual != null) ctx?.UnloadModule(kernelResidual.CUModule);
			smoother.cleanup();
			ctx?.Dispose();
			ctx = null;

			if (un != null) for (int i = 0; i <= un.GetUpperBound(0); i++) UtilsCU.disposeBuf(ref un[i]);
			if (res != null) for (int i = 0; i <= res.GetUpperBound(0); i++) UtilsCU.disposeBuf(ref res[i]);
			if (rhs != null) for (int i = 0; i <= rhs.GetUpperBound(0); i++) if (rhs[i] != fn) UtilsCU.disposeBuf(ref rhs[i]);
			UtilsCU.disposeBuf(ref fn);
			UtilsCU.disposeBuf(ref flagCU);
			un0 = null;
		}
	}
}

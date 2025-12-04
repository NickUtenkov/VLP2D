using ManagedCuda;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class ProgonkaSchemeCU<T> : Iterative1DScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>
	{
		protected T stepX2, stepY2, dt, eps;
		protected CudaDeviceVariable<T> unmCU;
		protected T[] alphaX, alphaY;//beta is placed to result
		protected int cXSegments, cYSegments;
		protected readonly bool bProgonkaFixedIters;
		protected Action<int> calculateIterationAlpha = null;

		protected CudaContext ctx;
		protected CudaKernel[] kernels;
		CudaKernel kernelEpsExceeded;
		protected object[][] args;
		object[] argsEpsExceeded;
		protected CudaDeviceVariable<T> inputCU, outputCU, fnCU, alphaXCU, alphaYCU;//fnCU unified memory
		CudaDeviceVariable<int> flagCU;//unified memory
		protected T[] un;
		bool unChanged = false;
		protected T _2 = T.CreateTruncating(2);

		public ProgonkaSchemeCU(int cXSegments1, int cYSegments1, T stepX, T stepY, T epsIn, bool bProgonkaFixedIters, Func<T, T, T> fKsi, int cudaDevice)
		{
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			cXSegments = cXSegments1;
			cYSegments = cYSegments1;

			dimX = cXSegments + 1;
			dimY = cYSegments + 1;

			un = new T[dimX * dimY];

			alphaX = new T[cXSegments];
			alphaY = new T[cYSegments];

			eps = epsIn;

			ctx = new CudaContext(cudaDevice);
			try
			{
				inputCU = new CudaDeviceVariable<T>(dimX * dimY);
				outputCU = new CudaDeviceVariable<T>(dimX * dimY);
				unmCU = new CudaDeviceVariable<T>(dimX * dimY);
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
			flagCU = new CudaDeviceVariable<int>(1);
			kernels = new CudaKernel[2];
			args = new object[2][];

			this.bProgonkaFixedIters = bProgonkaFixedIters;
			if (!bProgonkaFixedIters)
			{
				kernelEpsExceeded = UtilsCU.createEpsExceededKernel<T>(dimX, dimY, ctx);
				argsEpsExceeded = [inputCU.DevicePointer, outputCU.DevicePointer, flagCU.DevicePointer, 0, eps];
				UtilsCU.set2DKernelDims(kernelEpsExceeded, dimX - 2, dimY - 2);
			}

			if (fKsi != null)
			{
				T[] fnFloat = new T[dimX * dimY];//exterior points are not used; can't iterate on fnCU, throws exceptions
				GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => fnFloat[i * dimY + j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
				fnCU = fnFloat;
				fnFloat = null;
			}

			calculateOptimalTimeStep(stepX, stepY);
		}

		public T doIteration(int iter)
		{
			unChanged = true;
			calculateIterationAlpha?.Invoke(iter);

			setKernel0Arguments(iter);
			kernels[0].Run(args[0]);

			setKernel1Arguments(iter);
			kernels[1].Run(args[1]);

			T rc = T.Zero;
			if (bProgonkaFixedIters) rc = eps + eps;
			else rc = epsExceeded() ? eps + eps : T.Zero;

			UtilsSwap.swap(ref inputCU, ref outputCU);

			return rc;
		}

		void calculateOptimalTimeStep(T stepX, T stepY)
		{
			T a = stepX * T.CreateTruncating(cXSegments);
			T b = stepY * T.CreateTruncating(cYSegments);
			T a2 = a * a;
			T b2 = b * b;
			dt = T.Sqrt(stepX2 + stepY2) * T.Pow(T.One / a2 + T.One / b2, T.CreateTruncating(-0.5)) / T.Pi;//Kalitkin p.406
		}

		protected void calcAlpha(T bx, T by)
		{
			alphaX[0] = T.Zero;
			for (int i = 1; i < cXSegments; i++) alphaX[i] = T.One / (bx - alphaX[i - 1]);

			alphaY[0] = T.Zero;
			for (int i = 1; i < cYSegments; i++) alphaY[i] = T.One / (by - alphaY[i - 1]);

			alphaXCU = alphaX;
			alphaYCU = alphaY;
		}

		bool epsExceeded()
		{
			flagCU[0] = 0;

			kernelEpsExceeded.Run(argsEpsExceeded);

			return flagCU[0] == 1;
		}

		protected virtual void setKernel0Arguments(int iter)
		{
			args[0][0] = inputCU.DevicePointer;
		}

		protected virtual void setKernel1Arguments(int iter)
		{
			args[1][1] = outputCU.DevicePointer;
		}

		public override T[] getArray()
		{
			if (unChanged)
			{
				unChanged = false;

				inputCU.CopyToHost(un);
			}

			return un;
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
			inputCU.CopyToDevice(un);
			outputCU.CopyToDevice(inputCU);
			unmCU.CopyToDevice(inputCU);
		}

		public virtual int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public virtual void cleanup()
		{
			UtilsCU.disposeBuf(ref inputCU);
			UtilsCU.disposeBuf(ref outputCU);
			UtilsCU.disposeBuf(ref unmCU);
			UtilsCU.disposeBuf(ref fnCU);
			UtilsCU.disposeBuf(ref flagCU);

			if (kernels != null) ctx?.UnloadModule(kernels[0].CUModule);
			ctx?.Dispose();
			ctx = null;
		}
	}
}

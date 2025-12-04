using ManagedCuda;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class SimpleIterationSchemeCU<T> : Iterative1DScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T[] un;
		T[] tauk;//chebish
		T eps;
		bool isChebysh;
		int maxIters;
		CudaContext ctx;
		CudaKernel kernel, kernelEpsExceeded;
		CudaDeviceVariable<T> inputCU, outputCU, fnCU;//fnCU unified memory
		CudaDeviceVariable<int> flagCU;//unified memory
		object[] args, argsEpsExceeded;
		bool uuChanged = false;
		T _2 = T.CreateTruncating(2);

		public SimpleIterationSchemeCU(int cXSegments, int cYSegments, T stepX, T stepY, bool isChebyshIn, T epsIn, Func<T, T, T> fKsi, int cudaDevice)
		{
			dimX = cXSegments + 1;
			dimY = cYSegments + 1;

			T stepX2 = stepX * stepX;
			T stepY2 = stepY * stepY;
			eps = epsIn;
			isChebysh = isChebyshIn;
			bool equalSteps = T.Abs(stepX - stepY) < T.Min(stepX, stepY) / T.CreateTruncating(100);//less than one percent
			T tau = stepX2 * stepY2 / ((stepX2 + stepY2) * _2);// step2 / 4.0 if stepX = stepY

			ctx = new CudaContext(cudaDevice);
			try
			{
				un = new T[dimX * dimY];
				inputCU = new CudaDeviceVariable<T>(dimX * dimY);
				outputCU = new CudaDeviceVariable<T>(dimX * dimY);
				if (fKsi != null)
				{
					GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => un[i * dimY + j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
					fnCU = un;//instead of 'new CudaDeviceVariable' and CopyToDevice
				}
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
			int threadsPerBlockDim1 = 16;
			int threadsPerBlockDim2 = 16;
			SimpleIterationKernelsCU kernelCreator = new SimpleIterationKernelsCU(ctx, Utils.getTypeName<T>(), threadsPerBlockDim1 + 2, threadsPerBlockDim2 + 2);

			if (fKsi == null)
			{
				if (equalSteps)
				{
					if (isChebysh)
					{
						kernel = kernelCreator.createLaplaceEqualStepsKernel();
						kernel.SetConstantVariable("stepX2", stepX2);
						args = [inputCU.DevicePointer, outputCU.DevicePointer, tau];
					}
					else
					{
						kernel = kernelCreator.createLaplaceEqualStepsNoChebKernel();
						args = [inputCU.DevicePointer, outputCU.DevicePointer];
					}
				}
				else
				{
					kernel = kernelCreator.createLaplaceKernel();
					kernel.SetConstantVariable("stepX2", stepX2);
					kernel.SetConstantVariable("stepY2", stepY2);
					args = [inputCU.DevicePointer, outputCU.DevicePointer, tau];
				}
			}
			else
			{
				if (equalSteps)
				{
					if (isChebysh)
					{
						kernel = kernelCreator.createPoissonEqualStepsKernel();
						kernel.SetConstantVariable("stepX2", stepX2);
						args = [inputCU.DevicePointer, outputCU.DevicePointer, tau, fnCU.DevicePointer];
					}
					else
					{
						kernel = kernelCreator.createPoissonEqualStepsNoChebKernel();
						kernel.SetConstantVariable("stepX2", stepX2);
						args = [inputCU.DevicePointer, outputCU.DevicePointer, fnCU.DevicePointer];
					}
				}
				else
				{
					kernel = kernelCreator.createPoissonKernel();
					kernel.SetConstantVariable("stepX2", stepX2);
					kernel.SetConstantVariable("stepY2", stepY2);
					args = [inputCU.DevicePointer, outputCU.DevicePointer, tau, fnCU.DevicePointer];
				}
			}
			int upperX = dimX - 2;
			int upperY = dimY - 2;
			kernel.SetConstantVariable("dimY", dimY);
			kernel.SetConstantVariable("upperX", upperX);
			kernel.SetConstantVariable("upperY", upperY);
			int lastBlockSizeX = upperX % threadsPerBlockDim1;
			int lastBlockSizeY = upperY % threadsPerBlockDim2;
			kernel.SetConstantVariable("lastBlockSizeX", lastBlockSizeX);
			kernel.SetConstantVariable("lastBlockSizeY", lastBlockSizeY);

			UtilsCU.set2DKernelDims(kernel, upperX, upperY, threadsPerBlockDim1, threadsPerBlockDim2);

			if (isChebysh)
			{//[SNR] p.300
				T _4 = T.CreateTruncating(4);
				T arg1 = T.Pi / (_2 * T.CreateTruncating(cXSegments));
				T arg2 = T.Pi / (_2 * T.CreateTruncating(cYSegments));

				T sin1 = T.Sin(arg1);
				T sin2 = T.Sin(arg2);
				T gamma1 = (sin1 * sin1 / stepX2 + sin2 * sin2 / stepY2) * _4;//[SNR] p.300,(9)

				T cos1 = T.Cos(arg1);
				T cos2 = T.Cos(arg2);
				T gamma2 = (cos1 * cos1 / stepX2 + cos2 * cos2 / stepY2) * _4;//[SNR] p.300,(9)

				T ksi = gamma1 / gamma2;
				T tau0 = _2 / (gamma1 + gamma2);
				T ro0 = (T.One - ksi) / (T.One + ksi);
				T ksiRoot = T.Sqrt(ksi);
				T ro1 = (T.One - ksiRoot) / (T.One + ksiRoot);
				maxIters = (int)uint.CreateTruncating(T.Log(T.CreateTruncating(0.5) * eps) / T.Log(ro1));//[SNR] p.300,(7)
				if (maxIters <= 0) maxIters = 5;
				tauk = new T[maxIters];
				int[] cheb = UtilsChebysh.chebyshParams(maxIters);
				for (int iter = 0; iter < maxIters; iter++)
				{
					T tk = -T.Cos((T.Pi * T.CreateTruncating(cheb[iter])) / T.CreateTruncating(2 * maxIters));//maxIters is n, k = 1,2,...,n
					tauk[iter] = tau0 / (T.One + ro0 * tk);//[SNR] p.299,(6)
				}
			}
			else
			{
				kernelEpsExceeded = UtilsCU.createEpsExceededKernel<T>(dimX, dimY, ctx);
				flagCU = new CudaDeviceVariable<int>(1);
				argsEpsExceeded = [inputCU.DevicePointer, outputCU.DevicePointer, flagCU.DevicePointer, 0, eps];

				UtilsCU.set2DKernelDims(kernelEpsExceeded, upperX, upperY);
			}
		}

		~SimpleIterationSchemeCU()
		{
		}

		public T doIteration(int iter)
		{
			uuChanged = true;

			args[0] = inputCU.DevicePointer;//set again because pointers is swaped
			args[1] = outputCU.DevicePointer;
			if (isChebysh) args[2] = tauk[iter];

			kernel.Run(args);

			UtilsSwap.swap(ref inputCU, ref outputCU);

			T rc = default;
			if (isChebysh) rc = eps + eps;
			else rc = epsExceeded() ? eps + eps : T.Zero;

			return rc;
		}

		bool epsExceeded()
		{
			flagCU[0] = 0;

			kernelEpsExceeded.Run(argsEpsExceeded);

			return flagCU[0] == 1;
		}

		public override T[] getArray()
		{
			if (uuChanged)
			{
				uuChanged = false;
				inputCU.CopyToHost(un);
			}
			return un; 
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			inputCU.CopyToDevice(un);
			outputCU.CopyToDevice(inputCU);
		}

		public int maxIterations() { return maxIters; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			UtilsCU.disposeBuf(ref inputCU);
			UtilsCU.disposeBuf(ref outputCU);
			UtilsCU.disposeBuf(ref fnCU);
			UtilsCU.disposeBuf(ref flagCU);
			un = null;
			tauk = null;
			args = null;
			argsEpsExceeded = null;

			if (kernel != null) ctx?.UnloadKernel(kernel);
			if (kernelEpsExceeded != null) ctx?.UnloadKernel(kernelEpsExceeded);
			ctx?.Dispose();
			ctx = null;
		}

		public override IterationsKind iterationsKind()
		{
			return isChebysh ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}
	}
}

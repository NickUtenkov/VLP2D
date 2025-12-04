using ManagedCuda;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class RelaxationSchemeCU<T> : Iterative1DScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T[] un;
		readonly T eps;
		CudaContext ctx;
		CudaKernel kernel;
		CudaDeviceVariable<T> inoutCU, fnCU;//fnCU unified memory
		CudaDeviceVariable<int> flagCU;//unified memory
		object[] args;
		bool uuChanged = false;

		public RelaxationSchemeCU(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, bool isSeidel, T eps, int cudaDevice)
		{
			dimX = cXSegments + 1;
			dimY = cYSegments + 1;
			un = new T[dimX * dimY];

			this.eps = eps;
			T stepX2 = (stepX * stepX);
			T stepY2 = (stepY * stepY);
			T coef = T.Zero;
			bool equalSteps = T.Abs(stepX - stepY) < T.Min(stepX, stepY) / T.CreateTruncating(100);//less than one percent
			T _2 = T.CreateTruncating(2);
			T _4 = T.CreateTruncating(4);
			if (!equalSteps) coef = T.One / (_2 / stepX2 + _2 / stepY2);

			T sinX = T.Sin(T.Pi / (T.CreateTruncating(cXSegments) * _2));//[SNR] p.382(bottom)
			T sinY = T.Sin(T.Pi / (T.CreateTruncating(cYSegments) * _2));//[SNR] p.382(bottom)
			T sumStep2 = stepX2 + stepY2;
			T lyambdaMin = stepY2 * _2 / sumStep2 * sinX * sinX + stepX2 * _2 / sumStep2 * sinY * sinY;//[SNR] p.382(bottom)
			T omega = _2 / (T.One + T.Sqrt(lyambdaMin * (_2 - lyambdaMin)));//[SNR] p.379(14)
			T omegaCoef;
			if (equalSteps) omegaCoef = omega / _4;
			else omegaCoef = omega * coef;
			T oneMinusOmega = T.One - omega;

			ctx = new CudaContext(cudaDevice);
			try
			{
				inoutCU = new CudaDeviceVariable<T>(dimX * dimY);
				flagCU = new CudaDeviceVariable<int>(1);
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
			int threadsPerBlockDim1 = 16;
			int threadsPerBlockDim2 = 16;
			RelaxationKernelsCU kernelCreator = new RelaxationKernelsCU(ctx, Utils.getTypeName<T>(), threadsPerBlockDim1 + 2, threadsPerBlockDim2 * 2 + 2);

			if (fKsi == null)
			{
				args = [0, inoutCU.DevicePointer, flagCU.DevicePointer];
				if (equalSteps)
				{
					if (isSeidel) kernel = kernelCreator.createLaplaceEqualStepsSeidelKernel();
					else
					{
						kernel = kernelCreator.createLaplaceEqualStepsKernel();
						kernel.SetConstantVariable("omegaCoef", omegaCoef);
						kernel.SetConstantVariable("oneMinusOmega", oneMinusOmega);
					}
				}
				else
				{
					if (isSeidel)
					{
						kernel = kernelCreator.createLaplaceSeidelKernel();
						kernel.SetConstantVariable("coef", coef);
						kernel.SetConstantVariable("stepX2", stepX2);
						kernel.SetConstantVariable("stepY2", stepY2);
					}
					else
					{
						kernel = kernelCreator.createLaplaceKernel();
						kernel.SetConstantVariable("omegaCoef", omegaCoef);
						kernel.SetConstantVariable("oneMinusOmega", oneMinusOmega);
						kernel.SetConstantVariable("stepX2", stepX2);
						kernel.SetConstantVariable("stepY2", stepY2);
					}
				}
			}
			else
			{
				T[] fnFloat = new T[dimX * dimY];//exterior points are not used
				T premultiply = equalSteps ? stepX2 : T.One;
				GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => fnFloat[i * dimY + j] = premultiply * fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
				fnCU = fnFloat;

				args = [0, inoutCU.DevicePointer, fnCU.DevicePointer, flagCU.DevicePointer];
				if (equalSteps)
				{
					if (isSeidel) kernel = kernelCreator.createPoissonEqualStepsSeidelKernel();
					else
					{
						kernel = kernelCreator.createPoissonEqualStepsKernel();
						kernel.SetConstantVariable("omegaCoef", omegaCoef);
						kernel.SetConstantVariable("oneMinusOmega", oneMinusOmega);
					}
				}
				else
				{
					if (isSeidel)
					{
						kernel = kernelCreator.createPoissonSeidelKernel();
						kernel.SetConstantVariable("coef", coef);
						kernel.SetConstantVariable("stepX2", stepX2);
						kernel.SetConstantVariable("stepY2", stepY2);
					}
					else
					{
						kernel = kernelCreator.createPoissonKernel();
						kernel.SetConstantVariable("omegaCoef", omegaCoef);
						kernel.SetConstantVariable("oneMinusOmega", oneMinusOmega);
						kernel.SetConstantVariable("stepX2", stepX2);
						kernel.SetConstantVariable("stepY2", stepY2);
					}
				}
			}

			int upperX = dimX - 2;
			int upperY = dimY - 2;
			kernel.SetConstantVariable("dimY", dimY);
			kernel.SetConstantVariable("upperX", upperX);
			kernel.SetConstantVariable("upperY", upperY);
			kernel.SetConstantVariable("eps", eps);
			int kernelDimX = upperX;
			int kernelDimY = upperY / 2;
			int lastBlockSizeX = kernelDimX % threadsPerBlockDim1;
			int lastBlockSizeY = kernelDimY % threadsPerBlockDim2;
			kernel.SetConstantVariable("lastBlockSizeX", lastBlockSizeX);
			//kernel.SetConstantVariable("lastBlockSizeY", lastBlockSizeY);

			UtilsCU.set2DKernelDims(kernel, kernelDimX, kernelDimY, threadsPerBlockDim1, threadsPerBlockDim2);
		}

		~RelaxationSchemeCU()
		{
		}

		public T doIteration(int iter)
		{
			uuChanged = true;

			flagCU[0] = 0;
			for (int i = 0; i <= 1; i++)
			{
				args[0] = i;
				kernel.Run(args);
			}

			return flagCU[0] == 1 ? eps + eps : T.Zero;
		}

		public override T[] getArray()
		{
			if (uuChanged)
			{
				uuChanged = false;
				inoutCU.CopyToHost(un);
			}
			return un;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			inoutCU.CopyToDevice(un);
		}

		public int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			UtilsCU.disposeBuf(ref inoutCU);
			UtilsCU.disposeBuf(ref fnCU);
			UtilsCU.disposeBuf(ref flagCU);
			un = null;
			args = null;

			if (kernel != null) ctx?.UnloadKernel(kernel);
			ctx?.Dispose();
			ctx = null;
		}
	}
}

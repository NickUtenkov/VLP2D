
using Cloo;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class SimpleIterationSchemeOCL<T> : Iterative1DScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T tau, stepX2, stepY2, eps;
		T[] tauk;//chebish
		bool isChebysh;
		int maxIters;

		KernelOCL kernel, kernelEpsExceeded;
		BufferOCL<T> inputOCL, outputOCL, fnOCL;
		CommandQueueOCL commands;
		object[] argsK;
		T[] un;
		long[] gWorkSize = [0, 0], workSize2DInternalPoints, lWorkSize = [0, 0];
		long[] gWork2DOffset = [1, 1];
		bool uuChanged = false;
		int[] flag = [0];
		BufferOCL<int> flagOCL;

		public SimpleIterationSchemeOCL(int cXSegments, int cYSegments, T stepX, T stepY, bool isChebyshIn, T epsIn, Func<T, T, T> fKsi, PlatformOCL platform, DeviceOCL device)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			dimX = cXSegments + 1;
			dimY = cYSegments + 1;
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			eps = epsIn;
			isChebysh = isChebyshIn;
			bool equalSteps = T.Abs(stepX - stepY) < T.Min(stepX, stepY) / T.CreateTruncating(100);//less than one percent
			tau = stepX2 * stepY2 / ((stepX2 + stepY2) * T.CreateTruncating(2));// step2 / 4.0 if stepX = stepY

			commands = UtilsCL.createCommandQueue(platform, device, CommandQueueFlagsOCL.None);

			try
			{
				un = new T[dimX * dimY];
				inputOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, un.Length);
				outputOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, un.Length);
				if (fKsi != null)
				{
					GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => un[i * dimY + j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
					fnOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, un);
				}
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			int upperX = dimX - 2;
			int upperY = dimY - 2;
			workSize2DInternalPoints = new long[] { upperX, upperY };

			int groupSize = 16;
			lWorkSize[0] = groupSize;
			lWorkSize[1] = groupSize;
			int lastBlockSizeX = upperX % (int)lWorkSize[0];
			int lastBlockSizeY = upperY % (int)lWorkSize[1];

			SimpleIterationKernelsOCL kernelCreator = new SimpleIterationKernelsOCL(commands, Utils.getTypeName<T>(), dimX, dimY, (int)lWorkSize[0] + 2, (int)lWorkSize[1] + 2, lastBlockSizeX, lastBlockSizeY);

			if (fKsi == null)
			{
				if (equalSteps)
				{
					if (isChebysh)
					{
						kernel = kernelCreator.createLaplaceEqualStepsKernel();
						argsK = [inputOCL, outputOCL, tau, stepX2];
					}
					else
					{
						kernel = kernelCreator.createLaplaceEqualStepsNoChebKernel();
						argsK = [inputOCL, outputOCL];
					}
				}
				else
				{
					kernel = kernelCreator.createLaplaceKernel();
					argsK = [inputOCL, outputOCL, tau, stepX2, stepY2];
				}
			}
			else
			{
				if (equalSteps)
				{
					if (isChebysh)
					{
						kernel = kernelCreator.createPoissonEqualStepsKernel();
						argsK = [inputOCL, outputOCL, tau, fnOCL, stepX2];
					}
					else
					{
						kernel = kernelCreator.createPoissonEqualStepsNoChebKernel();
						argsK = [inputOCL, outputOCL, fnOCL, stepX2];
					}
				}
				else
				{
					kernel = kernelCreator.createPoissonKernel();
					argsK = [inputOCL, outputOCL, tau, fnOCL, stepX2, stepY2];
				}
			}
			UtilsCL.setKernelArguments<T>(kernel, argsK);

			gWorkSize[0] = ((upperX + lWorkSize[0] - 1) / lWorkSize[0]) * lWorkSize[0];
			gWorkSize[1] = ((upperY + lWorkSize[1] - 1) / lWorkSize[1]) * lWorkSize[1];

			if (isChebysh)
			{//[SNR] p.300
				T _4 = T.CreateTruncating(4);
				T arg1 = T.Pi / T.CreateTruncating(2.0 * cXSegments);
				T arg2 = T.Pi / T.CreateTruncating(2.0 * cYSegments);

				T sin1 = T.Sin(arg1);
				T sin2 = T.Sin(arg2);
				T gamma1 = (sin1 * sin1 / stepX2 + sin2 * sin2 / stepY2) * _4;//[SNR] p.300,(9)

				T cos1 = T.Cos(arg1);
				T cos2 = T.Cos(arg2);
				T gamma2 = (cos1 * cos1 / stepX2 + cos2 * cos2 / stepY2) * _4;//[SNR] p.300,(9)

				T ksi = gamma1 / gamma2;
				T tau0 = T.CreateTruncating(2) / (gamma1 + gamma2);
				T ro0 = (T.One - ksi) / (T.One + ksi);
				T ksiRoot = T.Sqrt(ksi);
				T ro1 = (T.One - ksiRoot) / (T.One + ksiRoot);
				maxIters = (int)uint.CreateTruncating(T.Log(eps * T.CreateTruncating(0.5)) / T.Log(ro1));//[SNR] p.300,(7)
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
				flagOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadWrite, 1);
				kernelEpsExceeded = UtilsCL.createProgramEpsExceeded(commands.Context, commands.Device, inputOCL, outputOCL, flagOCL, dimY, eps);
			}
		}

		~SimpleIterationSchemeOCL()
		{
		}

		public T doIteration(int iter)
		{
			uuChanged = true;

			kernel.SetMemoryArgument(0, inputOCL);//set again because pointers are swaped
			kernel.SetMemoryArgument(1, outputOCL);
			if (isChebysh) kernel.SetValueArgument(2, tauk[iter]);

			commands.Execute(kernel, gWork2DOffset, gWorkSize, lWorkSize, null);
			commands.Finish();

			UtilsSwap.swap(ref inputOCL, ref outputOCL);

			T rc = T.Zero;
			if (isChebysh) rc = eps + eps;
			else rc = epsExceeded() ? eps + eps : T.Zero;

			return rc;
		}

		bool epsExceeded()
		{
			flag[0] = 0;
			commands.WriteToBuffer(flag, flagOCL, true, null);

			commands.Execute(kernelEpsExceeded, gWork2DOffset, workSize2DInternalPoints, null, null);
			commands.Finish();

			commands.ReadFromBuffer(flagOCL, ref flag, true, null);
			return flag[0] == 1;
		}

		public override T[] getArray()
		{
			if (uuChanged)
			{
				uuChanged = false;

				commands.ReadFromBuffer(inputOCL, ref un, true, null);
			}

			return un;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			commands.WriteToBuffer(un, inputOCL, true, null);
			commands.WriteToBuffer(un, outputOCL, true, null);
		}

		public int maxIterations() { return maxIters; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			UtilsCL.disposeQC(ref commands);
			UtilsCL.disposeKP(ref kernel);
			UtilsCL.disposeKP(ref kernelEpsExceeded);
			UtilsCL.disposeBuf(ref inputOCL);
			UtilsCL.disposeBuf(ref outputOCL);
			UtilsCL.disposeBuf(ref fnOCL);
			UtilsCL.disposeBuf(ref flagOCL);
			un = null;
		}

		public override IterationsKind iterationsKind()
		{
			return isChebysh ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}
	}
}

using Cloo;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class RelaxationSchemeOCL<T> : Iterative1DScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		readonly T stepX2, stepY2, eps, coef;
		T omega, rJacobi2;
		bool isChebysh;
		T omegaCoef, oneMinusOmega;
		bool uuChanged = false;

		CommandQueueOCL commands;
		KernelOCL kernel;
		BufferOCL<T> un, fn;
		T[] un0;
		BufferOCL<int> flagOCL;
		int[] flag = { 0 };
		readonly long[] gWorkSize, lWorkSize = [0, 0];
		long[] gWork2DOffset = [1, 1];
		object[] argsK;
		T _4 = T.CreateTruncating(4);

		public RelaxationSchemeOCL(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, bool isSeidel, bool isChebyshIn, T eps, PlatformOCL platform, DeviceOCL device)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			dimX = cXSegments + 1;
			dimY = cYSegments + 1;
			un0 = new T[dimX * dimY];
			this.eps = eps;
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			T _2 = T.CreateTruncating(2);
			T _cXSegments = T.CreateTruncating(cXSegments);
			T _cYSegments = T.CreateTruncating(cYSegments);
			bool equalSteps = T.Abs(stepX - stepY) < T.Min(stepX, stepY) / T.CreateTruncating(100);//less than one percent
			if (!equalSteps) coef = T.One / (_2 / stepX2 + _2 / stepY2);
			isChebysh = !isSeidel && isChebyshIn;

			commands = UtilsCL.createCommandQueue(platform, device, CommandQueueFlagsOCL.None);

			try
			{
				un0 = new T[dimX * dimY];
				un = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, un0.Length);
				flagOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadWrite, 1);
				if (fKsi != null)
				{
					T premultiply = equalSteps ? stepX2 : T.One;
					GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => un0[i * dimY + j] = premultiply * fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
					fn = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, un0);
				}
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			if (!isChebysh)
			{
				T sinX = T.Sin(T.Pi / (_2 * _cXSegments));//[SNR] p.382(bottom)
				T sinY = T.Sin(T.Pi / (_2 * _cYSegments));//[SNR] p.382(bottom)
				T sumStep2 = stepX2 + stepY2;
				T lyambdaMin = _2 * stepY2 / sumStep2 * sinX * sinX + _2 * stepX2 / sumStep2 * sinY * sinY;//[SNR] p.382(bottom)
				omega = _2 / (T.One + T.Sqrt(lyambdaMin * (_2 - lyambdaMin)));//[SNR] p.379(14)
				if (equalSteps) omegaCoef = omega / _4;
				else omegaCoef = omega * coef;
				oneMinusOmega = T.One - omega;
			}
			else
			{
				omega = T.One;
				omegaCoef = omega / _4;
				oneMinusOmega = T.One - omega;
				omega = T.CreateTruncating(0.5);//for omega = 1.0 / (1.0 - rJacobi2 / 2.0) be omega = 1.0 / (1.0 - rJacobi2 * omega / 4.0) on first iteration(iter==0)

				T rJacobi = (T.Cos(T.Pi / _cXSegments) + T.Cos(T.Pi / _cYSegments)) / _2;//NumericalRecipesinC,891,(19.5.24); deltaX == deltaY(equal steps in X & Y directions)
				rJacobi2 = rJacobi * rJacobi;
			}

			int upperX = dimX - 2;
			int upperY = dimY - 2;
			int groupSize = 16;
			lWorkSize[0] = groupSize;
			lWorkSize[1] = groupSize;
			int lastBlockSizeX = upperX % (int)lWorkSize[0];
			int lastBlockSizeY = upperY % (int)lWorkSize[1];

			string strCondition = (typeof(T) == typeof(float) || typeof(T) == typeof(double)) ? "fabs(un[idx] - s[ind]) > eps" : "gt(fabs(sub_HH(un[idx], s[ind])), eps)";
			RelaxationKernelsOCL kernelCreator = new RelaxationKernelsOCL(commands, Utils.getTypeName<T>(), dimX, dimY, (int)lWorkSize[0] + 2, (int)lWorkSize[1] * 2 + 2, lastBlockSizeX, lastBlockSizeY, strCondition);

			gWorkSize = [((upperX + lWorkSize[0] - 1) / lWorkSize[0]) * lWorkSize[0], ((upperY + lWorkSize[1] - 1) / lWorkSize[1]) * lWorkSize[1]];

			if (fKsi == null)
			{
				if (equalSteps)
				{
					if (isSeidel)
					{
						kernel = kernelCreator.createLaplaceEqualStepsSeidelKernel();
						argsK = [0, un, eps, flagOCL];
					}
					else
					{
						kernel = kernelCreator.createLaplaceEqualStepsKernel();
						argsK = [0, un, omegaCoef, oneMinusOmega, eps, flagOCL];
					}
				}
				else
				{
					if (isSeidel)
					{
						kernel = kernelCreator.createLaplaceSeidelKernel();
						argsK = [0, un, coef, stepX2, stepY2, eps, flagOCL];
					}
					else
					{
						kernel = kernelCreator.createLaplaceKernel();
						argsK = [0, un, omegaCoef, oneMinusOmega, stepX2, stepY2, eps, flagOCL];
					}
				}
			}
			else
			{
				if (equalSteps)
				{
					if (isSeidel)
					{
						kernel = kernelCreator.createPoissonEqualStepsSeidelKernel();
						argsK = [0, un, fn, eps, flagOCL];
					}
					else
					{
						kernel = kernelCreator.createPoissonEqualStepsKernel();
						argsK = [0, un, omegaCoef, oneMinusOmega, fn, eps, flagOCL];
					}
				}
				else
				{
					if (isSeidel)
					{
						kernel = kernelCreator.createPoissonSeidelKernel();
						argsK = [0, un, fn, coef, stepX2, stepY2, eps, flagOCL];
					}
					else
					{
						kernel = kernelCreator.createPoissonKernel();
						argsK = [0, un, omegaCoef, oneMinusOmega, fn, stepX2, stepY2, eps, flagOCL];
					}
				}
			}
			UtilsCL.setKernelArguments<T>(kernel, argsK);
		}

		~RelaxationSchemeOCL()
		{
		}

		public T doIteration(int iter)
		{
			uuChanged = true;

			if (isChebysh)
			{
				omega = T.One / (T.One - rJacobi2 * omega / _4);
				omegaCoef = omega / _4;
				oneMinusOmega = T.One - omega;
			}

			flag[0] = 0;
			commands.WriteToBuffer(flag, flagOCL, true, null);
			for (int j = 0; j <= 1; j++)
			{
				if (isChebysh)
				{
					kernel.SetValueArgument(2, omegaCoef);
					kernel.SetValueArgument(3, oneMinusOmega);
				}
				kernel.SetValueArgument(0, j);
				commands.Execute(kernel, gWork2DOffset, gWorkSize, lWorkSize, null);
				commands.Finish();
			}
			commands.ReadFromBuffer(flagOCL, ref flag, true, null);

			return flag[0] == 1 ? eps + eps : T.Zero;
		}

		public override T[] getArray()
		{
			if (uuChanged)
			{
				uuChanged = false;

				commands.ReadFromBuffer(un, ref un0, true, null);
			}

			return un0;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			commands.WriteToBuffer(un0, un, true, null);
		}

		public int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			UtilsCL.disposeQC(ref commands);
			UtilsCL.disposeKP(ref kernel);
			UtilsCL.disposeBuf(ref un);
			UtilsCL.disposeBuf(ref fn);
			UtilsCL.disposeBuf(ref flagOCL);
		}
	}
}

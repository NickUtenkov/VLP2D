
using CLMathLibraries.CLFFT;
using Cloo;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using VLP2D.Common;
using static VLP2D.Common.UtilsElapsed;

namespace VLP2D.Model
{
	class VariablesSeparationSchemeProgonkaOCL<T> : VariablesSeparationSchemeOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, IMinMaxValue<T>, ILogarithmicFunctions<T>
	{
		IFFTOCL<T> fft2;//for use inside fftN2
		VariablesSeparationFFTN2OCL<T> fftN2;
		Dictionary<int, CLFFTPlan> plans;
		VariablesSeparationProgonkaOCL<T> progonka;
		int[] fftWorkSizes, progonkaWorkSizes;

		public VariablesSeparationSchemeProgonkaOCL(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, ParallelOptions optionsParallel, PlatformOCL platform, DeviceOCL device, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, stepX, stepY, fKsi, optionsParallel, platform, device, reportProgressIn)
		{
			allFFTN2WorkSize = dim1;//number of vectors of size FFTN2DataSize(see below)
			int allProgonkaWorkSize = dim2;//number of vectors of size progonkaDataSize(see below)

			long maxFloats = UtilsCL.maxMemoryAllocationSize(device) / Marshal.SizeOf(typeof(T));

			FFTN2RealInputSize = (dim2 + 1);//vector lenth for which FFT is used(' + 1' for FFT padding)
			int FFTN2ComplexOutputSize = (FFTN2RealInputSize / 2 + 1) * FFTConstant.sizeOfComplex;
			maxFFTN2Vectors = int.Min((int)maxFloats / FFTN2ComplexOutputSize, allFFTN2WorkSize);
			long bufFFTN2OutputCount = maxFFTN2Vectors * FFTN2ComplexOutputSize;
			int maxWorkSize = Math.Min(maxFFTN2Vectors, FFTOCL<T>.maxBatchSize(maxFloats * Marshal.SizeOf(typeof(T)), FFTN2RealInputSize));
			fftWorkSizes = Utils.calculateWorkSizes(maxWorkSize, allFFTN2WorkSize);

			int progonkaDataSize = dim1;//vector length for which progonka is used
			int maxProgonkaVectors = int.Min((int)maxFloats / progonkaDataSize, allProgonkaWorkSize);
			progonkaWorkSizes = Utils.calculateWorkSizes(maxProgonkaVectors, allProgonkaWorkSize);

			long bufProgonkaCount = progonkaDataSize * maxProgonkaVectors;

			long bufDataCount = long.Max(bufFFTN2OutputCount, bufProgonkaCount);
			try
			{
				data = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, bufDataCount);

				if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
				{
					plans = new Dictionary<int, CLFFTPlan>();
					fft2 = new FFTOCL<T>(commands, plans, N2, true);
				}
				else fft2 = new FFTLomontOCL<T>(commands, N2);

				fftN2 = new VariablesSeparationFFTN2OCL<T>(commands, fft2, dim2, data, un);

				progonka = new VariablesSeparationProgonkaOCL<T>(commands, data, un, progonkaDataSize, N2, stepX2, stepY2);
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
		}

		~VariablesSeparationSchemeProgonkaOCL()
		{
		}

		override public T doIteration(int iter)
		{
			float elapsed;
			initElapsedList();
			elapsed = getExecutedSeconds(stopWatchEL, () => initRigthHandSide(fKsi, stepX, stepY));//near border values are initialized in initTopBottomBorders, initLeftRightBorders in DirectOCLScheme
			listElapsedAdd("initRHS", elapsed);

			try
			{
				elapsed = getExecutedSeconds(stopWatchEL, () => fftN2.calculate(fftWorkSizes, T.One, showProgress, 60, areIterationsCanceled));//[SNR] p.195, (37), same as [SNR] p.192, (24)
				listElapsedAdd("FFTN2 1", elapsed);
				if (areIterationsCanceled()) return T.Zero;

				elapsed = getExecutedSeconds(stopWatchEL, () => progonka.calculate(progonkaWorkSizes, showProgress, 20, areIterationsCanceled));
				listElapsedAdd("progonka", elapsed);
				if (areIterationsCanceled()) return T.Zero;

				elapsed = getExecutedSeconds(stopWatchEL, () => fftN2.calculate(fftWorkSizes, T.CreateTruncating(2.0 / N2), showProgress, 20, areIterationsCanceled));//[SNR] p.195, (39), same as [SNR] p.192
				listElapsedAdd("FFTN2 2", elapsed);
			}
			catch (Exception ex)
			{
				Debug.WriteLine(ex.Message);
			}

			return base.doIteration(iter);
		}

		override public void cleanup()
		{
			fft2.cleanup();
			fftN2?.cleanup();
			fftN2 = null;
			progonka?.cleanup();
			progonka = null;
			if (plans != null) foreach (var plan in plans) plan.Value.Destroy();
			plans = null;
			base.cleanup();
		}

		override public string getElapsedInfo() { return timesElapsed(); }

	}
}

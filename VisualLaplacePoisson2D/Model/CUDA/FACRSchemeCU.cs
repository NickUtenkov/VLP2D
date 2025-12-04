using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	internal class FACRSchemeCU<T> : Direct1DNoBoundariesScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, IMinMaxValue<T>, IRootFunctions<T>, ILogarithmicFunctions<T>
	{
		CudaContext ctx;
		CudaDeviceVariable<T> unCU, fftData;

		readonly int N1, N2, ML, L;
		Func<T, T, T> fKsi;
		T stepX, stepY;
		protected Action<double> reportProgress;
		float curProgress;
		bool iterationsCanceled;
		FACRForwardStepsCU<T> stepsL;
		FACRFFTCU<T> fft;
		FACRProgonkaEvenCU<T> progonkaEven;
		FACRProgonkaOddCU<T> progonkaOdd;
		long maxFFTVectors;
		int allFFTWorkSize;
		float[] unShow;
		List<BitmapSource> lstBitmap;
		Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;

		public FACRSchemeCU(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, int paramL, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, ParallelOptions optionsParallel, Action<double> reportProgressIn, int cudaDevice)
			: base(cXSegments - 1, cYSegments - 1, stepX, stepY, optionsParallel)
		{
			ctx = new CudaContext(cudaDevice);
			this.fKsi = fKsi;
			this.stepX = stepX;
			this.stepY = stepY;
			this.lstBitmap = lstBitmap;
			this.fCreateBitmap = fCreateBitmap;

			N1 = cXSegments;
			N2 = cYSegments;//is 2^x
			L = paramL;
			ML = N2 >> L;

			reportProgress = reportProgressIn;
			curProgress = 0;

			try
			{
				unCU = new CudaDeviceVariable<T>(dim1 * dim2);
				if (lstBitmap != null) unShow = new float[dim1 * dim2];

				int memDivider = 2;//4 as in OpenCL
				ManagedCuda.BasicTypes.SizeT memFreeSize = ctx.GetFreeDeviceMemorySize();
				long maxFloats = (memFreeSize / memDivider) / Marshal.SizeOf(typeof(T));

				int allProgonkaWorkSizeEven = ML - 1;//number of vectors of size progonkaDataSize(see below)

				int progonkaDataSize = dim1;//vector length for which progonka is used

				allFFTWorkSize = dim1;//number of vectors of size FFTSizeWithPadding(see below)
				int FFTRealInputSize = ML;//vector lenth for which FFT is used
				int FFTComplexOutputSize = (FFTRealInputSize / 2 + 1) * FFTConstant.sizeOfComplex;//DFT output satisfies the “Hermitian” redundancy
				maxFFTVectors = Math.Min(maxFloats / FFTComplexOutputSize, allFFTWorkSize);
				long bufSizeFFT = maxFFTVectors * FFTComplexOutputSize;
				fftData = new CudaDeviceVariable<T>(bufSizeFFT);
				fft = new FACRFFTCU<T>(ctx, N2, L, unCU, fftData);

				progonkaEven = new FACRProgonkaEvenCU<T>(ctx, unCU, allProgonkaWorkSizeEven, progonkaDataSize, N2, L, stepX2, stepY2);
				if (L > 0)
				{
					stepsL = new FACRForwardStepsCU<T>(ctx, unCU, dim1, dim2, N2, L, stepY2 / stepX2);
					progonkaOdd = new FACRProgonkaOddCU<T>(ctx, unCU, progonkaDataSize, N2, L, stepX2, stepY2);
				}
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
		}

		public T doIteration(int iter)
		{
			float elapsed;
			initElapsedList();

			elapsed = getExecutedSeconds(stopWatchEL, () => initRigthHandSide(fKsi, stepX, stepY));//near border values are initialized in initTopBottomBorders, initLeftRightBorders in base class
			listElapsedAdd("initRHS", elapsed);

			elapsed = getExecutedSeconds(stopWatchEL, () => unCU.CopyToDevice(un));
			listElapsedAdd("CopyToDevice", elapsed);

			if (L > 0)
			{
				elapsed = getExecutedSeconds(stopWatchEL, () => stepsL.calculate(areIterationsCanceled));//[SNR] p.202, (19) and p.201, (15)
				listElapsedAdd(string.Format("Forward {0} step(s)", L), elapsed);
				stepsL.cleanup();
			}
			if (areIterationsCanceled()) return T.Zero;
			showProgress(10);

			elapsed = getExecutedSeconds(stopWatchEL, () => fft.calculate((int)maxFFTVectors, allFFTWorkSize, T.One, showProgress, 30, areIterationsCanceled));//[SNR] p.202 (20);
			listElapsedAdd("FFT 1", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => progonkaEven.calculate());
			listElapsedAdd("progonkaEven", elapsed);
			progonkaEven.cleanup();
			if (areIterationsCanceled()) return T.Zero;
			showProgress(20);

			elapsed = getExecutedSeconds(stopWatchEL, () => fft.calculate((int)maxFFTVectors, allFFTWorkSize, T.CreateTruncating(2.0 * (1 << L) / N2), showProgress, 30, areIterationsCanceled));//[SNR] p.203, (23);
			listElapsedAdd("FFT2", elapsed);
			fft.cleanup();
			if (areIterationsCanceled()) return T.Zero;
			if (unShow != null)
			{
				for (int i = 0; i < dim1; i++ )
				{
					for (int j = (1 << L) - 1; j < N2 - 1; j += 1 << L) unShow[i * dim2 + j] = float.CreateTruncating(unCU[i * dim2 + j]);
				}
				UtilsPict.addPicture(lstBitmap, true, null, new Adapter2D<float>(dim1, dim2, (m, k) => unShow[m * dim2 + k]), fCreateBitmap);
			}

			if (L > 0)
			{
				elapsed = getExecutedSeconds(stopWatchEL, () => progonkaOdd.calculate(areIterationsCanceled, dim1, unShow, lstBitmap, fCreateBitmap));
				listElapsedAdd("progonkaOdd", elapsed);
				progonkaOdd.cleanup();
			}
			if (areIterationsCanceled()) return T.Zero;
			showProgress(10);

			elapsed = getExecutedSeconds(stopWatchEL, () => unCU.CopyToHost(un));
			listElapsedAdd("CopyToHost", elapsed);

			return T.Zero;
		}

		void initRigthHandSide(Func<T, T, T> fKsi, T stepX, T stepY)
		{
			if (fKsi != null) iterate((i, j) => un[i * dim2 + j] += fKsi(stepX * T.CreateTruncating(i + 1), stepY * T.CreateTruncating(j + 1)));
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		protected void showProgress(float count)
		{
			curProgress += count;
			reportProgress((int)curProgress);
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
			if (unShow != null) iterate((i, j) => unShow[i * dim2 + j] = float.NaN);
		}

		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public void cleanup()
		{
			UtilsCU.disposeBuf(ref unCU);
			UtilsCU.disposeBuf(ref fftData);
			un = null;
			fft?.cleanup();//use below ctx
			stepsL?.cleanup();
			progonkaOdd?.cleanup();
			progonkaEven?.cleanup();
			ctx?.Dispose();
			ctx = null;
		}

		protected bool areIterationsCanceled()
		{
			return iterationsCanceled;
		}
	}
}


using Cloo;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class FACRSchemeOCL<T> : Direct1DNoBoundariesScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		readonly int N1, N2, ML, L;
		Func<T, T, T> fKsi;
		T stepX, stepY;
		Action<double> reportProgress;
		float curProgress;
		bool iterationsCanceled;
		List<BitmapSource> lstBitmap;
		readonly Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;

		CommandQueueOCL commands;

		BufferOCL<T> unOCL, data;
		FACRForwardStepsOCL<T> stepsL;
		FACRFFTOCL<T> fft;
		FACRProgonkaEvenOCL<T> progonkaEven;
		FACRProgonkaOddOCL<T> progonkaOdd;
		long maxFFTVectors;
		int allFFTWorkSize;
		T[] unShow;

		public FACRSchemeOCL(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, int paramL, ParallelOptions optionsParallel, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, PlatformOCL platform, DeviceOCL device, Action<double> reportProgressIn)
			: base(cXSegments - 1, cYSegments - 1, stepX, stepY, optionsParallel)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			N1 = cXSegments;
			N2 = cYSegments;//is 2^x
			L = paramL;
			ML = N2 >> L;
			this.fKsi = fKsi;
			this.stepX = stepX;
			this.stepY = stepY;

			reportProgress = reportProgressIn;
			curProgress = 0;
			this.lstBitmap = lstBitmap;
			this.fCreateBitmap = fCreateBitmap;

			commands = UtilsCL.createCommandQueue(platform, device, CommandQueueFlagsOCL.None);

			long maxFloats = UtilsCL.maxMemoryAllocationSize(device) / Marshal.SizeOf(typeof(T));
			int allProgonkaWorkSizeEven = ML - 1;//number of vectors of size progonkaDataSize(see below)

			int progonkaDataSize = dim1;//vector length for which progonka is used

			allFFTWorkSize = dim1;//number of vectors of size FFTSizeWithPadding(see below)
			int FFTRealInputSize = ML;//vector lenth for which FFT is used
			int FFTComplexOutputSize = (FFTRealInputSize / 2 + 1) * FFTConstant.sizeOfComplex;//DFT output satisfies the “Hermitian” redundancy
			maxFFTVectors = Math.Min(maxFloats / FFTComplexOutputSize, allFFTWorkSize);
			long bufSizeFFT = maxFFTVectors * FFTComplexOutputSize;

			try
			{
				unOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dim1 * dim2);//9490x9490 - max float buffer size(1/4 of all GPU memory)(on Intel HD Graphics 5000)
				if (lstBitmap != null) unShow = new T[dim1 * dim2];
				data = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, bufSizeFFT);//used only in FFT calculation, other buffers are used in FFT & progonka
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			fft = new FACRFFTOCL<T>(commands, N2, L, data, unOCL);

			progonkaEven = new FACRProgonkaEvenOCL<T>(commands, unOCL, allProgonkaWorkSizeEven, progonkaDataSize, N2, L, stepX2, stepY2);

			if (L > 0)
			{
				stepsL = new FACRForwardStepsOCL<T>(commands, unOCL, dim1, dim2, N2, L, stepY2 / stepX2);
				progonkaOdd = new FACRProgonkaOddOCL<T>(commands, unOCL, progonkaDataSize, N2, L, stepX2, stepY2);
			}
		}

		public T doIteration(int iter)
		{
			if (fKsi != null) initRigthHandSide(fKsi, stepX, stepY);//near border values are initialized in initTopBottomBorders, initLeftRightBorders in DirectOCLScheme

			commands.WriteToBuffer(un, unOCL, true, null);

			if (L > 0) stepsL.calculate(areIterationsCanceled);//[SNR] p.202, (19) and p.201, (15)
			if (areIterationsCanceled()) return T.Zero;
			showProgress(10);

			fft.calculate((int)maxFFTVectors, allFFTWorkSize, T.One, showProgress, 30, areIterationsCanceled);//[SNR] p.202 (20);
			if (areIterationsCanceled()) return T.Zero;

			progonkaEven.calculate();
			if (areIterationsCanceled()) return T.Zero;
			showProgress(20);

			fft.calculate((int)maxFFTVectors, allFFTWorkSize, T.CreateTruncating(2.0 * (1 << L) / N2), showProgress, 30, areIterationsCanceled);//[SNR] p.203, (23);
			if (areIterationsCanceled()) return T.Zero;
			if (unShow != null)
			{
				for (int i = 0; i < dim1; i++)
				{
					for (int j = (1 << L) - 1; j < N2 - 1; j += 1 << L) commands.ReadFromBuffer(unOCL, ref unShow, true, i * dim2 + j, i * dim2 + j, 1, null);// unShow[i * dim2 + j] = unCU[i * dim2 + j];
				}
				UtilsPict.addPicture(lstBitmap, true, null, new Adapter2D<float>(dim1, dim2, (m, k) => float.CreateTruncating(unShow[m * dim2 + k])), fCreateBitmap);
			}

			if (L > 0) progonkaOdd.calculate(areIterationsCanceled, dim1, unShow, lstBitmap, fCreateBitmap);
			if (areIterationsCanceled()) return T.Zero;
			showProgress(10);

			commands.ReadFromBuffer(unOCL, ref un, true, null);

			return T.Zero;
		}

		void initRigthHandSide(Func<T, T, T> fKsi, T stepX, T stepY)
		{
			iterate((i, j) => un[i * dim2 + j] += fKsi(stepX * T.CreateTruncating(i + 1), stepY * T.CreateTruncating(j + 1)));
		}

		protected void showProgress(float count)
		{
			curProgress += count;
			reportProgress((int)curProgress);
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
			if (unShow != null) iterate((i, j) => unShow[i * dim2 + j] = T.Zero / T.Zero);
		}

		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public void cleanup()
		{
			stepsL?.cleanup();
			stepsL = null;
			fft?.cleanup();
			fft = null;
			progonkaEven?.cleanup();
			progonkaEven = null;
			progonkaOdd?.cleanup();
			progonkaOdd = null;
			UtilsCL.disposeQC(ref commands);
			UtilsCL.disposeBuf(ref unOCL);
			UtilsCL.disposeBuf(ref data);
			un = null;
		}

		protected bool areIterationsCanceled()
		{
			return iterationsCanceled;
		}
	}
}

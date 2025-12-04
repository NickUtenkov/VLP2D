using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class CyclicReductionBaseSchemeCU<T> : Direct1DBoundariesScheme<T>, IScheme<T> where T : struct, INumber<T>, IMinMaxValue<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>
	{
		protected T stepX, stepY, bCoef;
		Func<T, T, T> fKsi;
		protected readonly int n;
		protected List<BitmapSource> lstBitmap;
		protected readonly Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;
		protected Action<double> reportProgress;
		protected int progressSteps, curProgress;
		bool iterationsCanceled;

		protected CudaContext ctx;
		protected CudaDeviceVariable<T> unCU;

		public CyclicReductionBaseSchemeCU(int cXSegments, int cYSegments, T stepXIn, T stepYIn, Func<T, T, T> fKsiIn, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn, int cudaDevice) :
			base(cXSegments + 1, cYSegments + 1, fKsiIn == null, lstBitmap != null)
		{
			stepX = stepXIn;
			stepY = stepYIn;
			bCoef = stepY * stepY / (stepX * stepX);//[SNR] p.145, (4), steps are reversed
			reportProgress = reportProgressIn;

			n = (int)uint.CreateTruncating(T.Log2(T.CreateTruncating(N1)));

			this.lstBitmap = lstBitmap;
			this.fCreateBitmap = fCreateBitmap;

			progressSteps = 0;
			curProgress = 0;

			fKsi = fKsiIn;

			ctx = new CudaContext(cudaDevice);

			try
			{
				unCU = new CudaDeviceVariable<T>((N1 + 1) * (N2 + 1));
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
		}

		virtual public T doIteration(int iter)
		{
			return T.Zero;
		}

		protected void initRigthHandSide(T[] fj)
		{
			if (fKsi != null)
			{
				T stepX2 = stepX * stepX;//steps are reversed
				int upper1 = N1;
				int upper2 = N2;
				GridIterator.iterate(upper1, upper2, (i, j) => fj[i * dim2 + j] = stepX2 * fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));//[SNR] p.123 (8)
			}
		}

		protected void transferBottomTopToInterior(T[] fj)
		{
			// p_j(0)  = F_j, [SNR] p.138 1) - not (1)
			T mult = T.One / bCoef;//steps are reversed
			Parallel.For(1, N1, (j) =>
			{//additional init of F_j; fi overline,[SNR] p.123 (8)
				fj[j * dim2 + 1] += mult * un[j * dim2 + 0];
				fj[j * dim2 + N2 - 1] += mult * un[j * dim2 + N2];
			});
		}

		protected void showProgress()
		{
			curProgress++;
			reportProgress(curProgress * 100.0 / progressSteps);
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			initRigthHandSide(un);
			transferBottomTopToInterior(un);

			hostBufferToDeviceBuffer();

			if (unShow != null)
			{
				T nan = T.Zero / T.Zero;
				GridIterator.iterate(0, N1 + 1, 0, N2 + 1, (i, j) => unShow[i * dim2 + j] = nan);
				fillBounds(unShow);

				void fillBounds(T[] uu)
				{
					Parallel.For(1, N1, GridIterator.optionsParallel, (i) =>
					{
						uu[i * dim2 + 0] = un[i * dim2 + 0];
						uu[i * dim2 + N2] = un[i * dim2 + N2];
					});

					Parallel.For(1, N2, GridIterator.optionsParallel, (j) =>
					{
						uu[0 * dim2 + j] = un[0 * dim2 + j];
						uu[N1 * dim2 + j] = un[N1 * dim2 + j];
					});

					uu[0 * dim2 + 0] = nan;
					uu[N1 * dim2 + 0] = nan;
					uu[0 * dim2 + N2] = nan;
					uu[N1 * dim2 + N2] = nan;
				}
			}
		}

		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public virtual void cleanup()
		{
			UtilsCU.disposeContext(ref ctx);
			UtilsCU.disposeBuf(ref unCU);
			un = null;
		}

		protected bool areIterationsCanceled()
		{
			return iterationsCanceled;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		protected T cosKL(int k, int l)
		{
			return T.Cos(T.Pi * T.CreateTruncating(2 * l - 1) / T.CreateTruncating(1 << k));
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		protected T diagElem(int k, int l)
		{
			return (T.One + bCoef * (T.One - cosKL(k, l))) * T.CreateTruncating(2.0);//[SNR] p.145, (4)
		}

		protected void fillArray(T[] ar, Func<int, int, T> func)
		{
			int idx = 0;
			for (int k = 1; k <= n; k++)
			{
				int m = 1 << (k - 1);
				for (int l = 1; l <= m; l++) // l = 1..2^{k-1}
				{
					ar[idx++] = func(k, l);
				}
			}
		}

		protected void deviceBufferToHostBuffer()
		{
			unCU.CopyToHost(un);
		}

		protected void hostBufferToDeviceBuffer()
		{
			unCU.CopyToDevice(un);
		}

		protected void addPicture(int idx1, int idx2, int step)
		{
			for (int i = idx1; i <= idx2; i += step)
			{
				long offsetSrcDstInBytes = (i * (N2 + 1) + 1) * Marshal.SizeOf(typeof(T));
				unCU.CopyToHost(unShow, offsetSrcDstInBytes, offsetSrcDstInBytes, (N2 + 1) * Marshal.SizeOf(typeof(T)));
			}
			UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (i, j) => float.CreateTruncating(unShow[i * dim2 + j])), fCreateBitmap);
		}
	}
}

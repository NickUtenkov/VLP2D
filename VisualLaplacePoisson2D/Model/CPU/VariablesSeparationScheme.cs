
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class VariablesSeparationScheme<T> : DirectJagged2Scheme<T>, IScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		protected T[][] fn;
		protected float[][] unShow;
		protected readonly ParallelOptions optionsParallel;
		protected List<BitmapSource> lstBitmap;
		protected readonly Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;
		protected Action<double> reportProgress;
		protected int progressSteps, curProgress;
		protected T stepX, stepY, stepX2, stepY2;
		protected Func<T, T, T> fKsi;
		protected FFTCalculator<T> fftN2;
		bool iterationsCanceled;
		protected int cCores;
		protected int rem1, remPict;
		protected Action<int> addPictureAction = null;

		public VariablesSeparationScheme(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsi, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(cXSegments + 1, cYSegments + 1, fKsi == null)
		{//N2 is 2^x, N1 can not be pow of 2(because fft for cXSegments is not used)
			fn = un;
			stepX = stepXIn;
			stepY = stepYIn;
			stepX2 = stepXIn * stepXIn;
			stepY2 = stepYIn * stepYIn;
			this.fKsi = fKsi;
			reportProgress = reportProgressIn;
			this.cCores = cCores;
			optionsParallel = new ParallelOptions() { MaxDegreeOfParallelism = cCores};

			this.lstBitmap = lstBitmap;
			this.fCreateBitmap = fCreateBitmap;
			if (lstBitmap != null)
			{
				unShow = new float[cXSegments + 1][];
				for (int i = 0; i < cXSegments + 1; i++) unShow[i] = new float[cYSegments + 1];
			}
			if (unShow != null) addPictureAction = (i) =>
			{
				for (int k = 1; k < N2; k++) unShow[i][k] = float.CreateTruncating(un[i][k]);
				if ((remPict > 0) && (i % remPict == 0)) UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (m, k) => unShow[m][k]), fCreateBitmap);
			};

			progressSteps = 0;
			curProgress = 0;

			fftN2 = new FFTCalculator<T>(cCores,N2);
		}

		protected void initRigthHandSide(Func<T, T, T> fKsi, T stepX, T stepY)
		{
			if (fKsi != null)
			{
				int upper1 = fn.GetUpperBound(0);
				int upper2 = fn[0].GetUpperBound(0);
				GridIterator.iterate(upper1, upper2, (i, j) => fn[i][j] += fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}
		}

		protected void transferBoundaryValuesToNearBoundaryNodes()
		{//[SNR] p.190, (19)
			Parallel.For(1, N2, (j) =>
			{
				fn[1][j] += un[0][j] / stepX2;
				fn[N1 - 1][j] += un[N1][j] / stepX2;
			});
			Parallel.For(1, N1, (i) =>
			{
				fn[i][1] += un[i][0] / stepY2;
				fn[i][N2 - 1] += un[i][N2] / stepY2;
			});
		}

		virtual public T doIteration(int iter)
		{
			return T.Zero;
		}

		protected void fftN2Calculate(T coef, Action<int> addPictureAction)
		{
			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
			{
				if (loopState.IsStopped) return;
				for (int i = 1 + core; i < N1; i += cCores)
				{
					fftN2.calculate(core, (k) => un[i][k], (k, val) => un[i][k] = coef * val);

					addPictureAction?.Invoke(i);
					if ((rem1 > 0) && (i % rem1 == 0)) showProgress();
					if (areIterationsCanceled())
					{
						loopState.Stop();
						return;
					}
				}
			});
		}

		protected void showProgress()
		{
			curProgress++;
			reportProgress(curProgress * 100.0 / progressSteps);
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
			if (unShow != null) GridIterator.iterateEdgesAndFillInternalPoints(N1 + 1, N2 + 1, (i, j) => unShow[i][j] = float.CreateTruncating(un[i][j]), (i, j) => unShow[i][j] = float.NaN);
		}

		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public override void cleanup()
		{
			fn = null;
			fftN2 = null;
			base.cleanup();
		}

		protected bool areIterationsCanceled()
		{
			return iterationsCanceled;
		}
	}
}

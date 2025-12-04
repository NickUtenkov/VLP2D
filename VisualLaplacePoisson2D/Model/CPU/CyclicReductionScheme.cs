#define MeetingProgonka

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class CyclicReductionScheme<T> : DirectJagged2Scheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		protected T[][] alfa;
		protected T stepX, stepY, bCoef;
		Func<T, T, T> fKsi;
		protected readonly bool isMultiThread;
		protected readonly int n;
		protected List<BitmapSource> lstBitmap;
		protected readonly Func<bool, UtilsPict.MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;
		protected float[][] unShow;
		protected readonly ParallelOptions optionsParallel;
		protected Action<double> reportProgress;
		protected int progressSteps, curProgress;
		bool iterationsCanceled;
		protected AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		protected int alfaUpperBound, progonkaUpperBound;
#if MeetingProgonka
		int midX;
#endif
		T _2 = T.CreateTruncating(2);

		public CyclicReductionScheme(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsiIn, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap,Action<double> reportProgressIn) :
			base(cXSegments + 1, cYSegments + 1, fKsiIn == null)
		{
			stepX = stepXIn;
			stepY = stepYIn;
			bCoef = stepY * stepY / (stepX * stepX);//[SNR] p.145, (4), steps are reversed
			reportProgress = reportProgressIn;
			isMultiThread = cCores > 1;
			optionsParallel = new ParallelOptions() { MaxDegreeOfParallelism = cCores };

			n = (int)uint.CreateTruncating(T.Log(T.CreateTruncating(N1), T.CreateTruncating(2)));//N1 is 2^x

			this.lstBitmap = lstBitmap;
			this.fCreateBitmap = fCreateBitmap;
			if (lstBitmap != null)
			{
				unShow = new float[cXSegments + 1][];
				for (int i = 0; i < cXSegments + 1; i++) unShow[i] = new float[cYSegments + 1];
			}

			progressSteps = 0;
			curProgress = 0;

			fKsi = fKsiIn;

			progonkaUpperBound = N2 - 2;
#if MeetingProgonka
			if ((N2 & 1) == 0) throw new System.Exception("CyclicReductionScheme N2 should be odd");
			midX = (N2 - 1) / 2;
			alfaUpperBound = midX - 1;
#else
			alfaUpperBound = progonkaUpperBound;
#endif
		}

		virtual public T doIteration(int iter)
		{
			return T.Zero;
		}

		protected void initRigthHandSide(T[][] fj)
		{
			T stepX2 = stepX * stepX;//steps are reversed
			if (fKsi != null) GridIterator.iterate(fj.GetUpperBound(0), fj[0].GetUpperBound(0), (i, j) => { fj[i][j] = stepX2 * fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)); });//[SNR] p.123 (8)
		}

		protected void transferBottomTopToInterior(T[][] fj)
		{
			T mult = T.One / bCoef;//steps are reversed
			Parallel.For(1, fj.GetUpperBound(0), (i) =>
			{//additional init of F_j; fi overline,[SNR] p.123 (8)
				fj[i][1] += mult * un[i][0];
				fj[i][N2 - 1] += mult * un[i][N2];
			});
		}

		protected void progonka(Func<int, T> rhs, T[] alf, T[] res)//[SNR] p.145 bottom
		{//beta array is not used - it is placed to res; res skips zero index(boundary values)
			int kUp = alf.GetUpperBound(0);
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			int ind(int idx) => (idx < kUp) ? idx : kUp;
#if MeetingProgonka
			int U = progonkaUpperBound;
			for (int i = 0; i <= alfaUpperBound; i++)//alfaUpperBound = midX - 1
			{
				res[i + 0 + 1] = (bCoef * rhs(i + 0 + 1) + ((i != 0) ? res[i + 0 - 0] : T.Zero)) * alf[ind(i)];//SNR p.77(12,formula 2)
				res[U - i + 1] = (bCoef * rhs(U - i + 1) + ((i != 0) ? res[U - i + 2] : T.Zero)) * alf[ind(i)];//SNR p.77(12,formula 4)
			}

			res[midX + 1] = (res[alfaUpperBound + 1 + 1] + alf[ind(alfaUpperBound)] * res[alfaUpperBound + 1]) / (T.One - alf[ind(alfaUpperBound)] * alf[ind(alfaUpperBound)]);//SNR p.77(13,formula 3)

			for (int i = midX - 1; i >= 0; i--) res[i + 1] += alf[ind(i + 0)] * res[i + 1 + 1];//SNR p.77(13,formula 1)
			for (int i = midX + 1; i <= U; i++) res[i + 1] += alf[ind(U - i)] * res[i - 1 + 1];//SNR p.77(13,formula 2)
#else
			res[1] = bCoef * rhs(1) * alf[ind(0)];//[SNR] p.145 bottom
			for (int i = 1; i <= progonkaUpperBound; i++) res[i + 1] = (bCoef * rhs(i + 1) + res[i]) * alf[ind(i)];

			for (int i = progonkaUpperBound - 1; i >= 0; i--) res[i + 1] += alf[ind(i)] * res[i + 2];
#endif
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

		protected bool areIterationsCanceled()
		{
			return iterationsCanceled;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		protected T cosKL(int k, int l)
		{
			return T.Cos(T.Pi * T.CreateTruncating(2 * l - 1) / T.CreateTruncating(1 << k));
		}

		protected void fillAlphaArrays()
		{
			int alfaLng = 0;
			int idx = 0;
			for (int k = 1; k <= n; k++)
			{
				int m = 1 << (k - 1);
				for (int l = 1; l <= m; l++) // l = 1..2^{k-1}
				{
					T diag = diagElem(k, l);
					int kUp = αCC.upperBound(diag, alfaUpperBound);
					alfaLng += kUp + 1;
					alfa[idx] = new T[kUp + 1];
					calcAlpha(diag, alfa[idx], kUp);
					idx++;
				}
			}
			double perc = alfaLng * 100.0 / ((N1 - 1) * (N2 - 2));
			Debug.WriteLine(string.Format("Alfa percentage {0:0.000}%, count {1} of {2}", perc, alfaLng, (N1 - 1) * (N2 - 2)));
			void calcAlpha(T diagElem, T[] alfa, int k)
			{
				alfa[0] = T.One / diagElem;//[SNR] p.145 bottom
				for (int i = 1; i <= k; i++) alfa[i] = T.One / (diagElem - alfa[i - 1]);//[SNR] p.145 bottom
			}

			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			T diagElem(int k, int l)
			{
				return bCoef * (-cosKL(k, l) * _2 + _2) + _2;//[SNR] p.145, (4)
			}
		}

		public override void cleanup()
		{
			unShow = null;
			base.cleanup();
		}
	}
}

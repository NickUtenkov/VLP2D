using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class MatrixProgonkaScheme<T> : DirectJagged2Scheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{//modified block elimination method
		T[][] alfa,vk, vkSum;
		T stepX, stepY, cBase, subsupra;//subsupra - subdiagonal,supradiagonal
		Func<T, T, T> fKsi;
		readonly int alfaUpperBound, progonkaUpperBound;
		bool iterationsCanceled;
		protected List<BitmapSource> lstBitmap;
		protected readonly Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;
		float[][] unShow;
		Action<double> reportProgress;
		int progressSteps, curProgress;
		readonly ParallelOptions optionsParallel;
		int cCores;
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		T _2 = T.CreateTruncating(2);

		public MatrixProgonkaScheme(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsi, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(cXSegments + 1, cYSegments + 1, fKsi == null)
		{
			optionsParallel = new ParallelOptions() { MaxDegreeOfParallelism = cCores };

			vk = new T[cCores][];
			vkSum = new T[cCores][];
			for (int i = 0; i < cCores; i++)
			{
				vk[i] = new T[cYSegments - 1];
				vkSum[i] = new T[cYSegments - 1];
			}
			alfa = new T[cCores][];
			for (int i = 0; i < cCores; i++) alfa[i] = new T[N2 - 1];

			alfaUpperBound = alfa[0].GetUpperBound(0);
			progonkaUpperBound = vk[0].GetUpperBound(0);

			stepX = stepXIn;
			stepY = stepYIn;
			subsupra = stepX * stepX / (stepY * stepY);//[SNR] p.106, (4), steps are reversed

			this.fKsi = fKsi;

			this.lstBitmap = lstBitmap;
			this.fCreateBitmap = fCreateBitmap;
			if (lstBitmap != null)
			{
				unShow = new float[cXSegments + 1][];
				for (int i = 0; i < cXSegments + 1; i++) unShow[i] = new float[cYSegments + 1];
			}

			reportProgress = reportProgressIn;
			progressSteps = 100;//2 loops by 50
			curProgress = 0;

			cBase = (T.One + subsupra) * _2;//[SNR] p.106, (4)

			this.cCores = cCores;
		}

		public T doIteration(int iter)
		{
			int rem1 = N1 / 50;
			int rem2 = N2 / 10;

			initFj(un);

			for (int i = 1; i < N1; i++)
			{
				Parallel.For(0, optionsParallel.MaxDegreeOfParallelism, optionsParallel, k => { for (int j = 0; j < N2 - 1; j++) vkSum[k][j] = T.Zero; });
				for (int j = 1; j < N2; j++) un[i][j] += un[i - 1][j];//Fj + beta(j)(beta is zero-based),[SNR] p.117 (40)
				matrixAjMultiplyVectorUsingProgonka(i, 0);//[SNR] p.117 (40)

				if (iterationsCanceled) return T.Zero;
				if ((rem1 > 0) && (i % rem1 == 0)) showProgress();
			}

			for (int i = N1 - 1; i > 0; i--)
			{
				Parallel.For(0, optionsParallel.MaxDegreeOfParallelism, optionsParallel, k => { for (int j = 0; j < N2 - 1; j++) vkSum[k][j] = (k == 0) ? un[i][j + 1] : T.Zero; });
				matrixAjMultiplyVectorUsingProgonka(i, 1);//[SNR] p.117 (41)
				if (unShow != null) for (int j = 1; j < N2; j++) unShow[i][j] = float.CreateTruncating(un[i][j]);

				if (iterationsCanceled) return T.Zero;
				if ((rem1 > 0) && (i % rem1 == 0)) showProgress();
				if ((rem2 > 0) && (i % rem2 == 0)) UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (m, k) => unShow[m][k]), fCreateBitmap);
			}

			return T.Zero;
		}

		void matrixAjMultiplyVectorUsingProgonka(int resultRow,int deltaRow)
		{
			int div = (cCores > 1) ? cCores : 0;
			int jOrI = resultRow + 1;
			int rightHandSideRow = resultRow + deltaRow;
			T _jOrI = T.CreateTruncating(jOrI);

			Parallel.For(0, cCores, optionsParallel, (j, loopState) =>
			{
				if (loopState.IsStopped) return;
				for (int k = j + 1; k < jOrI; k += cCores)
				{
					(T sin, T cos) = T.SinCosPi(T.CreateTruncating(k) / _jOrI);
					T diagElem = (cBase - cos * _2) / subsupra;//[SNR] p.119, (49), left side
					T akj = sin * sin * _2 / _jOrI / subsupra;//[SNR] p.119, (48)
					int kUp = calcAlpha(alfa[j], diagElem);
					progonka((i) => akj * un[rightHandSideRow][i + 1], vk[j], alfa[j], diagElem, kUp);//akj * res[...] is [SNR] p.119, (49), right side
					for (int i = 0; i < N2 - 1; i++) vkSum[j][i] += vk[j][i];

					if (iterationsCanceled)
					{
						loopState.Stop();
						return;
					}
				}
			});
			if (iterationsCanceled) return;

			for (int k = 1; k < optionsParallel.MaxDegreeOfParallelism; k++) for (int j = 0; j <= progonkaUpperBound; j++) vkSum[0][j] += vkSum[k][j];
			for (int j = 0; j <= progonkaUpperBound; j++) un[resultRow][j + 1] = vkSum[0][j];
		}

		void initFj(T[][] fj)
		{
			T stepX2 = stepX * stepX;//steps are reversed
			int upper1 = fj.GetUpperBound(0);
			int upper2 = fj[0].GetUpperBound(0);
			if (fKsi != null) GridIterator.iterate(upper1, upper2, (i, j) => { fj[i][j] = stepX2 * fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)); });//[SNR] p.105(middle), analog [SNR] p.123 (8)
			else GridIterator.iterate(upper1, upper2, (i, j) => { fj[i][j] = T.Zero; });//reset to zero initial iteration

			Parallel.For(1, fj.GetUpperBound(0), (i) =>
			{//additional init of F_j; fi overline,[SNR] p.105(middle), analog [SNR] p.123 (8)
				fj[i][1] += subsupra * un[i][0];
				fj[i][N2 - 1] += subsupra * un[i][N2];
			});
		}

		int calcAlpha(T[] alfa, T diagElem)
		{
			int k = αCC.upperBound(diagElem, alfaUpperBound);
			alfa[0] = T.One / diagElem;//[SNR] p.75 (7)
			for (int i = 1; i <= k; i++) alfa[i] = T.One / (diagElem - alfa[i - 1]);//[SNR] p.75 (7)
			return k;
		}

		void progonka(Func<int, T> rhs, T[] res, T[] alfa, T diagElem, int k)
		{//beta array is not used - it is placed to res
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			int ind(int idx) => (idx < k) ? idx : k;
			res[0] = rhs(0) / diagElem;//[SNR] p.75 (8)
			for (int i = 1; i <= progonkaUpperBound; i++) res[i] = (rhs(i) + res[i - 1]) * alfa[ind(i)];//[SNR] p.75 (8)

			for (int i = progonkaUpperBound - 1; i >= 0; i--) res[i] += alfa[ind(i)] * res[i + 1];
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			if (unShow != null) GridIterator.iterateEdgesAndFillInternalPoints(N1 + 1, N2 + 1, (i, j) => unShow[i][j] = float.CreateTruncating(un[i][j]), (i, j) => unShow[i][j] = float.NaN);
		}

		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public override void cleanup()
		{
		}

		void showProgress()
		{
			curProgress++;
			reportProgress((curProgress * 100.0 / progressSteps));
		}
	}
}

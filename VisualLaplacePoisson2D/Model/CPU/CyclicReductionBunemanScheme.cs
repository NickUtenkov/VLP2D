using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class CyclicReductionBunemanScheme<T> : CyclicReductionScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{//Buneman variant 1 method
		T[][] q,p,v;
		readonly int iteratorUpperBound;
		protected int[][] matrixOrder;
		T _2 = T.CreateTruncating(2);

		public CyclicReductionBunemanScheme(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsi, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, stepXIn, stepYIn, cCores, fKsi, lstBitmap0, fCreateBitmap, reportProgressIn)
		{
			//only odd 1st indexes are used in math algorithm, so dimension of 1st index can be half size & use pIndex()
			p = new T[(N1 >> 1) - 1][];//value p[x,0] is not used - for progonka second index always add 1 for res argument
			for (int i = 0; i < (N1 >> 1) - 1; i++) p[i] = new T[N2];
			//originally(in math description) q & v are (N1 - 1)*(N2 - 1) size, here they are interior part of un, which is (N1 + 1)*(N2 + 1) size
			//so they are used with shifted by 1 indeces
			q = un;
			v = un;

			iteratorUpperBound = progonkaUpperBound + 1;

			matrixOrder = new int[n - 1][];//size == log₂(N1)*(N1-1)
			for (int i = 0; i < n - 1; i++) matrixOrder[i] = UtilsChebysh.reductionParams(i + 1);//[SNR] p.143-144

			alfa = new T[N1 - 1][];

			progressSteps = (n - 1) + n;//n - 1 - direct steps, n - reverse steps
		}

		override public T doIteration(int iter)
		{//Q = 5*N2*N1*Log(N1,2) + 5*N2*N1;N2 can be == N1
			int cCores = optionsParallel.MaxDegreeOfParallelism;
			initElapsedList();

			float elapsed = getExecutedSeconds(() => { fillAlphaArrays(); initRigthHandSide(q); transferBottomTopToInterior(q); });// step 1: k = 0, q_j(0)  = F_j, [SNR] p.141 1) - not (1)
			listElapsedAdd("RHS, transfer", elapsed);

			elapsed = getExecutedSeconds(() => forwardWay(cCores));
			listElapsedAdd("forwardWay", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(() => reverseWay(cCores));
			listElapsedAdd("reverseWay", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		void forwardWay(int cCores)
		{
			// step 2: k = 1
			Parallel.For(1, N1 / 2, optionsParallel, idx =>
			{
				int j = 2 * idx;
				T[] pp = p[idx - 1];//idx - 1 == pIndex(j)
				T[] qq = q[j];
				progonka((i) => qq[i], alfa[0], pp);
				for (int i = 1; i <= iteratorUpperBound; i++) q[j][i] = pp[i] * _2 + q[j - 1][i] + q[j + 1][i];//[SNR] p.141, (37)
			});
			showProgress();
			if (areIterationsCanceled()) return;

			for (int k = 2; k <= n - 1; k++)// step 3  k = 2,3,...n-1
			{
				int _2ᵏ = 1 << k;
				int allVectors = N1 / _2ᵏ - 1;//1·2ᵏ, 2·2ᵏ, 3·2ᵏ, ..., (N1 - 2ᵏ); == 1,2,3,...,(N1 / 2ᵏ - 1) · 2ᵏ, [SNR] p.141, (38)
				int m = 1 << (k - 1);//2ᵏ⁻¹
				GridIterator.iterateWithIndeces(allVectors, _2ᵏ, _2ᵏ, iteratorUpperBound, optionsParallel, (j, i) => v[j][i] = q[j][i] + p[pIndex(j - m)][i] + p[pIndex(j + m)][i]);//[SNR] p.141, (38)

				Parallel.For(0, Math.Min(allVectors, cCores), optionsParallel, core =>
				{
					for (int j = core + 1; j <= allVectors; j += cCores)
					{
						T[] vv = v[j * _2ᵏ];
						for (int l = 1; l <= m; l++) progonka((i) => vv[i], alfa[m - 2 + matrixOrder[k - 2][l - 1]], vv);
					}
				});

				GridIterator.iterateWithIndeces(allVectors, _2ᵏ, _2ᵏ, iteratorUpperBound, optionsParallel, (j, i) =>
				{
					T[] pp = p[pIndex(j)];
					pp[i] += v[j][i];//[SNR] p.142, (40)
					q[j][i] = pp[i] * _2 + q[j - m][i] + q[j + m][i];//[SNR] p.142, (40)
				});

				showProgress();
				if (areIterationsCanceled()) return;
			}
		}

		void reverseWay(int cCores)
		{
			for (int k = n; k >= 2; k--)// reverse steps
			{
				int m = 1 << (k - 1);//2ᵏ⁻¹
				int _2ᵏ = m << 1;
				int allVectors = N1 / _2ᵏ;//1·2ᵏ⁻¹, 3·2ᵏ⁻¹, 5·2ᵏ⁻¹, ..., (N1 - 2ᵏ⁻¹); == 1,3,5,...,(N1 / 2ᵏ⁻¹ - 1) · 2ᵏ⁻¹, [SNR] p.142, (41), (43)
				GridIterator.iterateWithIndeces(allVectors, m, _2ᵏ, iteratorUpperBound, optionsParallel, (j, i) => v[j][i] = q[j][i] + un[j - m][i] + un[j + m][i]);//[SNR] p.142, (41)

				Parallel.For(0, Math.Min(allVectors, cCores), optionsParallel, core =>
				{
					for (int j = core; j < allVectors; j += cCores)
					{
						T[] vv = v[m + j * _2ᵏ];
						for (int l = 1; l <= m; l++) progonka((i) => vv[i], alfa[m - 2 + matrixOrder[k - 2][l - 1]], vv);
					}
				});

				GridIterator.iterateWithIndeces(allVectors, m, _2ᵏ, iteratorUpperBound, optionsParallel, (j, i) => un[j][i] = p[pIndex(j)][i] + v[j][i]);//[SNR] p.142, (43)
				if (unShow != null) GridIterator.iterateWithIndeces(allVectors, m, _2ᵏ, iteratorUpperBound, optionsParallel, (j, i) => unShow[j][i] = float.CreateTruncating(un[j][i]));

				if (unShow != null) UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (i, j) => unShow[i][j]), fCreateBitmap);
				else UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (i, j) => float.CreateTruncating(un[i][j])), fCreateBitmap);
				showProgress();
				if (areIterationsCanceled()) return;
			}
			// k = 1
			Parallel.For(0, N1 / 2, optionsParallel, idx =>
			{
				int row = 1 + idx * 2;
				Func<int, T> rhs = (i) => q[row][i] + un[row - 1][i] + un[row + 1][i];
				progonka(rhs, alfa[0], un[row]);
			});//[SNR] p.142, (44)

			showProgress();
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		int pIndex(int idx) => (idx >> 1) - 1;//== idx / 2 - 1

		public override void cleanup()
		{
			q = null;
			p = null;
			v = null;
			alfa = null;
			base.cleanup();
		}
	}
}

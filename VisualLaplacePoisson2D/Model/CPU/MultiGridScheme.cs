
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class MultiGridScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T[,] fn;
		T[][,] u0, rhs, res;//right hand side, residual
		T[] stepX, stepY;
		MultiGridSlidingIteration<T> smoother;
		readonly int nLevels;
		T eps;
		bool iterationsCanceled;
		T _025 = T.CreateTruncating(0.25);
		T _05 = T.CreateTruncating(0.5);
		T _2 = T.CreateTruncating(2);
		T _4 = T.CreateTruncating(4);
		const int countSmoothingIterations = 3;//if 1 then no convergence in rectangle cases, if 2 then precision is less 10^3 times

		public MultiGridScheme(int cXSegments, int cYSegments, T stepXIn, T stepYIn, Func<T, T, T> fKsi, T eps)
		{
			if (fKsi != null)
			{
				fn = new T[cXSegments + 1, cYSegments + 1];//exterior points are not used
				GridIterator.iterate(cXSegments, cYSegments, (i, j) => fn[i, j] = -fKsi(stepXIn * T.CreateTruncating(i), stepYIn * T.CreateTruncating(j)));
			}

			this.eps = eps;

			int nLevelsX = (int)Math.Log(cXSegments, 2);
			int nLevelsY = (int)Math.Log(cYSegments, 2);
			nLevels = Math.Min(nLevelsX, nLevelsY);

			u0 = new T[nLevels][,];
			rhs = new T[nLevels][,];
			res = new T[nLevels][,];
			stepX = new T[nLevels];
			stepY = new T[nLevels];
			smoother = new MultiGridSlidingIteration<T>();

			int cSegsX = cXSegments;//is 2^n
			int cSegsY = cYSegments;//is 2^n
			T lngX = stepXIn * T.CreateTruncating(cSegsX);
			T lngY = stepYIn * T.CreateTruncating(cSegsY);
			for (int i = 0; i < nLevels; i++)
			{
				u0[i] = new T[cSegsX + 1, cSegsY + 1];
				rhs[i] = (i == 0) && (fn != null) ? fn : new T[cSegsX + 1, cSegsY + 1];
				res[i] = (i < nLevels - 1) ? new T[cSegsX + 1, cSegsY + 1] : null;
				stepX[i] = lngX / T.CreateTruncating(cSegsX);
				stepY[i] = lngY / T.CreateTruncating(cSegsY);
				smoother.addParameters(u0[i], (i == 0) ? fn : rhs[i], stepX[i], stepY[i]);

				cSegsX /= 2;
				cSegsY /= 2;
			}
			un0 = u0[0];
		}

		public T doIteration(int iter)
		{
			VCycle(0);
			//FMG();//can use instead of call VCycle(0)
			if (iterationsCanceled) return T.Zero;

			T rc = GridIterator.iterateUntilCondition(u0[0].GetUpperBound(0), u0[0].GetUpperBound(1), (i, j) => T.Abs(u0[0][i, j] - res[0][i, j]) > eps) ? T.One : T.Zero;
			return rc;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
		}

		public int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { iterationsCanceled = true; }

		public void cleanup()
		{
			un0 = null;
			fn = null;
			u0 = null;
			rhs = null;
			res = null;
			stepX = null;
			stepY = null;
			smoother = null;
		}

		void VCycle(int startLevel)
		{
			for (int level = startLevel; level <= nLevels - 2; level++)
			{
				smoother.doIterations(level, countSmoothingIterations);
				if (iterationsCanceled) return;
				residual(level);//calc res on level
				restrictResidual(level);//calc rhs on (level + 1)
				GridIterator.iterateWithEdges(u0[level + 1].GetUpperBound(0) + 1, u0[level + 1].GetUpperBound(1) + 1, (i, j) => u0[level + 1][i, j] = T.Zero);
			}
			if (iterationsCanceled) return;
			smoother.doIterations(nLevels - 1, countSmoothingIterations);
			if (iterationsCanceled) return;
			for (int level = nLevels - 2; level >= startLevel; level--)
			{
				//strict but not same as with GPU;if (level == 0) GridIterator.iterate(res[0].GetUpperBound(0), res[0].GetUpperBound(1), (i, j) => res[0][i, j] = u0[0][i, j]);//will be compare with(after this function)
				interpolate(level);
				//for same with GPU variants
				if (level == 0) GridIterator.iterate(res[0].GetUpperBound(0), res[0].GetUpperBound(1), (i, j) => res[0][i, j] = u0[0][i, j]);//will be compare with(after this function)
				smoother.doIterations(level, countSmoothingIterations);
				if (iterationsCanceled) return;
			}
		}

		void FMG()
		{
			for (int level = 0; level <= nLevels - 2; level++)
			{
				residual(level);
				restrictResidual(level);
				GridIterator.iterateWithEdges(u0[level + 1].GetUpperBound(0) + 1, u0[level + 1].GetUpperBound(1) + 1, (i, j) => u0[level + 1][i, j] = T.Zero);
			}
			smoother.doIterations(nLevels - 1, 1);
			for (int level = nLevels - 2; level >= 0; level--)
			{
				interpolate(level);
				VCycle(level);
			}
			VCycle(0);
		}

		void residual(int level)
		{
			T[,] rh = rhs[level];
			T[,] un = u0[level];
			T[,] re = res[level];

			T stepX2 = stepX[level] * stepX[level];
			T stepY2 = stepY[level] * stepY[level];
			GridIterator.iterate(re.GetUpperBound(0), re.GetUpperBound(1), (i, j) => re[i, j] = rh[i, j] - UtilsOpLap.operatorLaplaceXY(un, i, j, stepX2, stepY2, _2));
		}

		void restrictResidual(int level)//Restriction
		{
			T[,] re = res[level + 0];
			T[,] rh = rhs[level + 1];
#if !Div16
			T oneDiv8 = T.One / T.CreateTruncating(8);
#else
			T oneDiv16 = T.One / T.CreateTruncating(16);
#endif

			Action<int, int> calcRHS = (i, j) =>
			{
				int _2i = i + i;
				int _2j = j + j;
#if !Div16
				rh[i, j] = oneDiv8 * (re[_2i, _2j] * _4 + (re[_2i + 1, _2j] + re[_2i - 1, _2j] + re[_2i, _2j + 1] + re[_2i, _2j - 1]));
#else
				rh[i, j] = oneDiv16 * (_4 * re[fi, fj] +
					_2 * (re[fi + 1, fj] + re[fi - 1, fj] + re[fi, fj + 1] + re[fi, fj - 1]) +
					T.One * (re[fi + 1, fj + 1] + re[fi - 1, fj - 1] + re[fi - 1, fj + 1] + re[fi + 1, fj - 1]));
#endif
				//rh[i, j] = _05 * re[fi, fj];//+- some epsilon can be used
			};
			GridIterator.iterate(rh.GetUpperBound(0), rh.GetUpperBound(1), (i, j) => calcRHS(i, j));
		}

		void interpolate(int level)//Prolongation
		{
			T[,] uc = u0[level + 1];//coarse
			T[,] uf = u0[level];//fine

			Action<int, int> interpol = (i, j) =>
			{
				int _2i = i + i;
				int _2j = j + j;
				if (i > 0 && j > 0) uf[_2i + 0, _2j + 0] += uc[i, j];
				if (j > 0) uf[_2i + 1, _2j + 0] += (uc[i, j] + uc[i + 1, j]) * _05;
				if (i > 0) uf[_2i + 0, _2j + 1] += (uc[i, j] + uc[i, j + 1]) * _05;
				uf[_2i + 1, _2j + 1] += (uc[i, j] + uc[i, j + 1] + uc[i + 1, j] + uc[i + 1, j + 1]) * _025;
			};
			GridIterator.iterate(0, uc.GetUpperBound(0), 0, uc.GetUpperBound(1), (i, j) => interpol(i, j));
		}
	}
}

using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class ConjugateGradientScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T stepX2, stepY2, stepXY, eps;
		T[,] un1;
		T[,] fn;
		Action<int,int> initRk;

		T[,] rk, zk;
		T[] columnSum;
		T ak, bk;
		int upper1, upper2;
		T _2 = T.CreateTruncating(2);

		public ConjugateGradientScheme(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, T eps)
		{
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			stepXY = stepX * stepY;
			this.eps = eps;

			un0 = new T[cXSegments + 1, cYSegments + 1];
			un1 = new T[cXSegments + 1, cYSegments + 1];
			upper1 = un0.GetUpperBound(0);
			upper2 = un0.GetUpperBound(1);

			rk = new T[cXSegments + 1, cYSegments + 1];
			zk = new T[cXSegments + 1, cYSegments + 1];
			columnSum = new T[cXSegments + 1];
			if (fKsi == null) initRk = initRkLap;
			else
			{
				initRk = initRkPoi;

				fn = new T[upper1 + 1, upper2 + 1];//exterior points are not used
				GridIterator.iterate(upper1, upper2, (i, j) => fn[i, j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}

			void initRkLap(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2);
			void initRkPoi(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2) - fn[i, j];//fn is NOT multiplied by step2 
		}

		T rk2(int i, int j) => rk[i, j] * rk[i, j];
		void funcYk(int i, int j) => un1[i, j] = un0[i, j] + ak * zk[i, j];
		void funcRk(int i, int j) => rk[i, j] = rk[i, j] - ak * UtilsOpLap.operatorLaplaceXY(zk, i, j, stepX2, stepY2, _2);
		void funcZk(int i, int j) => zk[i, j] = rk[i, j] + bk * zk[i, j];

		public T doIteration(int iter)
		{//https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D1%81%D0%BE%D0%BF%D1%80%D1%8F%D0%B6%D1%91%D0%BD%D0%BD%D1%8B%D1%85_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BE%D0%B2_(%D0%B4%D0%BB%D1%8F_%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D1%8F_%D0%A1%D0%9B%D0%90%D0%A3)
		 //https://en.wikipedia.org/wiki/Conjugate_gradient_method
			T rkScalarProduct = GridIterator.scalarProduct(rk, rk2, columnSum);

			T AzkzkScalarProduct = GridIterator.scalarProduct(zk, (i, j) => zk[i, j] * UtilsOpLap.operatorLaplaceXY(zk, i, j, stepX2, stepY2, _2), columnSum);

			ak = rkScalarProduct / AzkzkScalarProduct;
			T deltaMax = GridIterator.iterateForMaxWithEps(un1.GetUpperBound(0), un1.GetUpperBound(1), funcYk, (i, j) => T.Abs(un0[i, j] - un1[i, j]), eps);

			GridIterator.iterate(upper1, upper2, funcRk);

			T rkScalarProductPrev = rkScalarProduct;
			rkScalarProduct = GridIterator.scalarProduct(rk, rk2, columnSum);
			bk = rkScalarProduct / rkScalarProductPrev;

			GridIterator.iterate(upper1, upper2, funcZk);
			UtilsSwap.swap(ref un0, ref un1);

			return deltaMax;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			UtilsBorders.copyLeftRightValues(un0, un1);
			UtilsBorders.copyTopBottomValues(un0, un1);

			GridIterator.iterate(upper1, upper2, initRk);

			void initZk(int i, int j) => zk[i, j] = rk[i, j];
			GridIterator.iterate(upper1, upper2, initZk);
		}

		public int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			un0 = null;
			un1 = null;
			fn = null;

			rk = null;
			zk = null;
			columnSum = null;
		}
	}
}

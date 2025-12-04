using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class MinimumResidualScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T stepX2, stepY2, stepXY, eps;
		T[,] un1;
		T[,] fn;
		Action<int, int> funcFk;

		T[,] rk;
		T[] columnSum;
		T tauk;
		T _2 = T.CreateTruncating(2);
		T _4 = T.CreateTruncating(4);

		public MinimumResidualScheme(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, T eps)
		{
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			stepXY = stepX * stepY;

			un0 = new T[cXSegments + 1, cYSegments + 1];
			un1 = new T[cXSegments + 1, cYSegments + 1];

			rk = new T[cXSegments + 1, cYSegments + 1];
			columnSum = new T[cXSegments + 1];
			if (fKsi == null) funcFk = funcFkLap;
			else
			{
				funcFk = funcFkPoi;

				fn = new T[un0.GetUpperBound(0) + 1, un0.GetUpperBound(1) + 1];//exterior points are not used
				GridIterator.iterate(fn.GetUpperBound(0), fn.GetUpperBound(1), (i, j) => fn[i, j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}
			this.eps = eps;
		}

		void funcFkLap(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2);
		void funcFkPoi(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2) - fn[i, j];//fn is NOT multiplied by step2 
		void funcYk(int i, int j) => un1[i, j] = un0[i, j] - tauk * rk[i, j];

		public T doIteration(int iter)
		{
			GridIterator.iterate(rk.GetUpperBound(0), rk.GetUpperBound(1), funcFk);

			T ArkScalarProduct = -GridIterator.scalarProduct(rk, (i, j) => rk[i, j] * UtilsOpLap.operatorLaplace(rk, i, j, _4), columnSum);//NOT divide,multiply by stepX,stepY

			T ArkArkScalarProduct = GridIterator.scalarProduct(rk, (i, j) => UtilsOpLap.operatorLaplace(rk, i, j, _4) * UtilsOpLap.operatorLaplace(rk, i, j, _4), columnSum);//NOT divide by stepX,stepY(two times)
			ArkArkScalarProduct /= stepXY;

			tauk = ArkScalarProduct / ArkArkScalarProduct;//used in funcYk
			T rc = GridIterator.iterateForMaxWithEps(un1.GetUpperBound(0), un1.GetUpperBound(1), funcYk, (i, j) => T.Abs(un0[i, j] - un1[i, j]), eps);
			UtilsSwap.swap(ref un0, ref un1);
			return rc;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			UtilsBorders.copyLeftRightValues(un0, un1);
			UtilsBorders.copyTopBottomValues(un0, un1);
		}

		public int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			un0 = null;
			un1 = null;

			rk = null;
			columnSum = null;
			fn = null;
		}
	}
}

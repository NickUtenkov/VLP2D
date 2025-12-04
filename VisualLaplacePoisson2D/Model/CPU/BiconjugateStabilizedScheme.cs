using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class BiconjugateStabilizedScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T stepX2, stepY2;
		T[,] un1;
		T[,] fn;

		T[,] rk, r0, pk, sk, vk, tk;
		T rok, ak, wk;
		T[] columnSum;
		T r0r0ScalarProduct;
		Action<int, int> initRk;
		int upper1, upper2;
		T _2 = T.CreateTruncating(2);

		public BiconjugateStabilizedScheme(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi)
		{
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;

			un0 = new T[cXSegments + 1, cYSegments + 1];
			un1 = new T[cXSegments + 1, cYSegments + 1];

			rk = new T[cXSegments + 1, cYSegments + 1];
			r0 = new T[cXSegments + 1, cYSegments + 1];
			pk = new T[cXSegments + 1, cYSegments + 1];
			sk = new T[cXSegments + 1, cYSegments + 1];
			vk = new T[cXSegments + 1, cYSegments + 1];
			tk = new T[cXSegments + 1, cYSegments + 1];
			columnSum = new T[cXSegments + 1];

			upper1 = un0.GetUpperBound(0);
			upper2 = un0.GetUpperBound(1);

			if (fKsi == null) initRk = initRkLap;
			else
			{
				initRk = initRkPoi;

				fn = new T[upper1 + 1, upper2 + 1];//exterior points are not used
				if (fKsi != null) GridIterator.iterate(upper1, upper2, (i, j) => fn[i, j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}

			void initRkLap(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2);
			void initRkPoi(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2) - fn[i, j];//fn is NOT multiplied by step2 

			rok = ak = wk = T.One;

			void initVk(int i, int j) => vk[i, j] = T.Zero;
			GridIterator.iterate(upper1, upper2, initVk);

			void initPk(int i, int j) => pk[i, j] = T.Zero;
			GridIterator.iterate(upper1, upper2, initPk);
		}

		void funcVk(int i, int j) => vk[i, j] = UtilsOpLap.operatorLaplaceXY(pk, i, j, stepX2, stepY2, _2);
		void funcSk(int i, int j) => sk[i, j] = rk[i, j] - ak * vk[i, j];
		void funcTk(int i, int j) => tk[i, j] = UtilsOpLap.operatorLaplaceXY(sk, i, j, stepX2, stepY2, _2);
		void funcYk(int i, int j) => un1[i, j] = un0[i, j] + wk * sk[i, j] + ak * pk[i, j];
		void funcRk(int i, int j) => rk[i, j] = sk[i, j] - wk * tk[i, j];

		public T doIteration(int iter)//BiCGStab
		{//https://ru.wikipedia.org/wiki/%D0%A1%D1%82%D0%B0%D0%B1%D0%B8%D0%BB%D0%B8%D0%B7%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D1%8B%D0%B9_%D0%BC%D0%B5%D1%82%D0%BE%D0%B4_%D0%B1%D0%B8%D1%81%D0%BE%D0%BF%D1%80%D1%8F%D0%B6%D1%91%D0%BD%D0%BD%D1%8B%D1%85_%D0%B3%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BE%D0%B2
		 //https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
			T rokPrev = rok;
			rok = GridIterator.scalarProduct(r0, (i, j) => r0[i, j] * rk[i, j], columnSum);

			T bk = (rok / rokPrev) * (ak / wk);

			GridIterator.iterate(upper1, upper2, (i, j) => pk[i, j] = rk[i, j] + bk * (pk[i, j] - wk * vk[i, j]));

			GridIterator.iterate(upper1, upper2, funcVk);

			T r0vkScalarProduct = GridIterator.scalarProduct(r0, (i, j) => r0[i, j] * vk[i, j], columnSum);
			ak = rok / r0vkScalarProduct;

			GridIterator.iterate(upper1, upper2, funcSk);

			GridIterator.iterate(upper1, upper2, funcTk);

			T tkskScalarProduct = GridIterator.scalarProduct(tk, (i, j) => tk[i, j] * sk[i, j], columnSum);
			T tktkScalarProduct = GridIterator.scalarProduct(tk, (i, j) => tk[i, j] * tk[i, j], columnSum);
			wk = tkskScalarProduct / tktkScalarProduct;

			GridIterator.iterate(upper1, upper2, funcYk);

			GridIterator.iterate(upper1, upper2, funcRk);

			T rkrkScalarProduct = GridIterator.scalarProduct(rk, (i, j) => rk[i, j] * rk[i, j], columnSum);
			UtilsSwap.swap(ref un0, ref un1);
			return T.Sqrt(rkrkScalarProduct);
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			UtilsBorders.copyLeftRightValues(un0, un1);
			UtilsBorders.copyTopBottomValues(un0, un1);

			GridIterator.iterate(upper1, upper2, initRk);

			void initr0(int i, int j) => r0[i, j] = rk[i, j];
			GridIterator.iterate(upper1, upper2, initr0);
			r0r0ScalarProduct = GridIterator.scalarProduct(r0, (i, j) => r0[i, j] * r0[i, j], columnSum);
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
			r0 = null;
			pk = null;
			sk = null;
			vk = null;
			tk = null;
			columnSum = null;
		}
	}
}

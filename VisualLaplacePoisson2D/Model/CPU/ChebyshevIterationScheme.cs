using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class ChebyshevIterationScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T stepX2, stepY2, eps;
		T[,] un1;
		T[,] fn;
		int upper1, upper2;

		T d, c, alpha, beta;
		T[,] rk, pk;
		Action<int, int> initRk, funcRk;
		T _05 = T.CreateTruncating(0.5);
		T _2 = T.CreateTruncating(2);
		T _4 = T.CreateTruncating(4);

		public ChebyshevIterationScheme(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, T eps)
		{
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			this.eps = eps;

			un0 = new T[cXSegments + 1, cYSegments + 1];
			un1 = new T[cXSegments + 1, cYSegments + 1];
			upper1 = un0.GetUpperBound(0);
			upper2 = un0.GetUpperBound(1);

			T piHalf = T.Pi / _2;
			T _cXSegments = T.CreateTruncating(cXSegments);
			T sinX = T.Sin(piHalf / _cXSegments);
			T deltaMinX = sinX * sinX * _4 / stepX2;//[SNR] p.441, at middle
			T cosX = T.Cos(piHalf / _cXSegments);
			T deltaMaxX = cosX * cosX * _4 / stepX2;//[SNR] p.441, at middle

			T _cYSegments = T.CreateTruncating(cYSegments);
			T sinY = T.Sin(piHalf / _cYSegments);
			T deltaMinY = sinY * sinY * _4 / stepY2;//[SNR] p.441, at middle
			T cosY = T.Cos(piHalf / _cYSegments);
			T deltaMaxY = cosY * cosY * _4 / stepY2;//[SNR] p.441, at middle

			d = (deltaMaxX + deltaMaxY + deltaMinX + deltaMinY) / _2;
			c = (deltaMaxX + deltaMaxY - deltaMinX - deltaMinY) / _2;
			alpha = _2 / d;

			rk = new T[cXSegments + 1, cYSegments + 1];
			pk = new T[cXSegments + 1, cYSegments + 1];

			if (fKsi == null) initRk = initRkLap;
			else initRk = initRkPoi;

			if (fKsi != null)
			{
				fn = new T[upper1 + 1, upper2 + 1];//exterior points are not used
				GridIterator.iterate(upper1, upper2, (i, j) => fn[i, j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}

			void initRkLap(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2);
			void initRkPoi(int i, int j) => rk[i, j] = fn[i, j] - UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2);//fn is NOT multiplied by step2 

			if (fKsi == null) funcRk = funcRkLap;
			else funcRk = funcRkPoi;
		}

		void funcXk(int i, int j) => un1[i, j] = un0[i, j] + alpha * pk[i, j];
		void funcRkLap(int i, int j) => rk[i, j] = UtilsOpLap.operatorLaplaceXY(un1, i, j, stepX2, stepY2, _2);
		void funcRkPoi(int i, int j) => rk[i, j] = fn[i, j] + UtilsOpLap.operatorLaplaceXY(un1, i, j, stepX2, stepY2, _2);//?! fn is NOT multiplied by step2 
		void funcPk(int i, int j) => pk[i, j] = rk[i, j] + beta * pk[i, j];

		public T doIteration(int iter)
		{//http://crecs.ru/ru/numlabs/help/SLAE/index.html
			T deltaMax = GridIterator.iterateForMaxWithEps(un1.GetUpperBound(0), un1.GetUpperBound(1), funcXk, (i, j) => T.Abs(un0[i, j] - un1[i, j]), eps);

			GridIterator.iterate(upper1, upper2, funcRk);

			if (iter == 0)
			{
				T val = (c * alpha);
				beta = val * val * _05;
			}
			else
			{
				T val = (c * alpha) / _2;
				beta = val * val;
			}

			alpha = T.One / (d - beta / alpha);

			GridIterator.iterate(upper1, upper2, funcPk);
			UtilsSwap.swap(ref un0, ref un1);

			return deltaMax;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			UtilsBorders.copyLeftRightValues(un0, un1);
			UtilsBorders.copyTopBottomValues(un0, un1);

			GridIterator.iterate(upper1, upper2, initRk);

			void initPk(int i, int j) => pk[i, j] = rk[i, j];
			GridIterator.iterate(upper1, upper2, initPk);
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
			pk = null;
		}
	}
}

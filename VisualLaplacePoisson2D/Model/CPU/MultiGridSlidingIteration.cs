using System;
using System.Collections.Generic;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class MultiGridSlidingIteration<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>//, IMultiplyOperators<T, double, T>
	{
		T[,] un, fn;
		T stepX2, stepY2, coef;
		Action<int, int> func;
		Action<T[,] , Action<int, int> , Action<int, int> > iterate;
		T _025 = T.CreateTruncating(0.25);
		List<Params> listParams = new List<Params>();
		class Params
		{
			public T[,] un, fn;
			public T stepX2, stepY2;
			public Action<int, int> func;
			public Action<T[,], Action<int, int>, Action<int, int>> iterate;
			public T coef;
		}

		public MultiGridSlidingIteration()
		{
		}

		public void addParameters(T[,] un, T[,] fn, T stepX, T stepY)
		{
			Params pars = new Params();
			listParams.Add(pars);

			pars.un = un;
			pars.fn = fn;

			pars.stepX2 = stepX * stepX;
			pars.stepY2 = stepY * stepY;

			if (un.GetUpperBound(0) > 64) pars.iterate = GridIterator.iterateRedBlack;
			else pars.iterate = GridIterator.iterateRedBlackSequent;

			bool equalSteps = T.Abs(stepX - stepY) < T.CreateTruncating(1E-15);
			if (equalSteps) pars.func = (fn == null) ? funcLapEqualSteps : funcPoiEqualSteps;
			else
			{
				T _2 = T.CreateTruncating(2.0);
				pars.coef = T.One / (_2 / pars.stepX2 + _2 / pars.stepY2);
				pars.func = (fn == null) ? funcLap : funcPoi;
			}
		}

		void funcLapEqualSteps(int i, int j) => un[i, j] = (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1]) * _025;//fn is null
		void funcPoiEqualSteps(int i, int j) => un[i, j] = (un[i - 1, j] + un[i + 1, j] + un[i, j - 1] + un[i, j + 1] - fn[i, j] * stepX2) * _025;//minus before fn differs SimpleIterationScheme,SlidingIterationScheme

		void funcLap(int i, int j) => un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2);//fn is null
		void funcPoi(int i, int j) => un[i, j] = coef * ((un[i - 1, j] + un[i + 1, j]) / stepX2 + (un[i, j - 1] + un[i, j + 1]) / stepY2 - fn[i, j]);//minus before fn differs SimpleIterationScheme,SlidingIterationScheme

		public void doIterations(int level, int nIterations)//O(h^2);sliding iteration method Gauss-Seidel;sequential shift method;Libman method
		{
			un = listParams[level].un;
			fn = listParams[level].fn;
			stepX2 = listParams[level].stepX2;
			stepY2 = listParams[level].stepY2;
			coef = listParams[level].coef;
			iterate = listParams[level].iterate;
			func = listParams[level].func;

			for (int i = 0; i < nIterations; i++) iterate(un, func, func);
		}
	}
}

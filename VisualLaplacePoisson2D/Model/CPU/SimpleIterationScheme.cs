
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class SimpleIterationScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IExponentialFunctions<T>
	{
		readonly T tau, stepX2, stepY2, eps;
		T[] tauk;//chebish
		T[,] un1, fn;
		bool isChebysh;
		int maxIters, upper1, upper2;
		Action<int, int> actionCalculate;
		T tau1;
		readonly T _025 = T.CreateTruncating(0.25);
		readonly T _05 = T.CreateTruncating(0.5);
		readonly T _2 = T.CreateTruncating(2);
		readonly T _4 = T.CreateTruncating(4);

		public SimpleIterationScheme(int cXSegments, int cYSegments, T stepX, T stepY, bool isChebyshIn, T epsIn, Func<T, T, T> fKsi)
		{
			un0 = new T[cXSegments + 1, cYSegments + 1];
			un1 = new T[cXSegments + 1, cYSegments + 1];
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			upper1 = cXSegments;
			upper2 = cYSegments;
			eps = epsIn;
			isChebysh = isChebyshIn;
			bool equalSteps = T.Abs(stepX - stepY) < T.Min(stepX , stepY) / T.CreateTruncating(100);//less than one percent
			tau = stepX2 * stepY2 / ((stepX2 + stepY2) * _2);// step2 / 4.0 if stepX = stepY
			if (fKsi == null)
			{
				if (equalSteps)
				{
					if (isChebysh) actionCalculate = funcLapEqualSteps;
					else actionCalculate = funcLapEqualStepsNoCheb;
				}
				else actionCalculate = funcLap;
			}
			else
			{
				if (equalSteps)
				{
					if (isChebysh) actionCalculate = funcPoiEqualSteps;
					else actionCalculate = funcPoiEqualStepsNoCheb;
				}
				else actionCalculate = funcPoi;
				fn = new T[upper1 + 1, upper2 + 1];//exterior points are not used
				GridIterator.iterate(upper1, upper2, (i, j) => fn[i, j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}

			if (isChebysh)
			{//[SNR] p.300
				T arg1 = T.Pi / (_2 * T.CreateTruncating(cXSegments));
				T arg2 = T.Pi / (_2 * T.CreateTruncating(cYSegments));

				T sin1 = T.Sin(arg1);
				T sin2 = T.Sin(arg2);
				T gamma1 = (sin1 * sin1 / stepX2 + sin2 * sin2 / stepY2) * _4;//[SNR] p.300,(9)

				T cos1 = T.Cos(arg1);
				T cos2 = T.Cos(arg2);
				T gamma2 = (cos1 * cos1 / stepX2 + cos2 * cos2 / stepY2) * _4;//[SNR] p.300,(9)

				T ksi = gamma1 / gamma2;
				T tau0 = _2 / (gamma1 + gamma2);
				T ro0 = (T.One - ksi) / (T.One + ksi);
				T ksiRoot = T.Sqrt(ksi);
				T ro1 = (T.One - ksiRoot) / (T.One + ksiRoot);
				maxIters = (int)uint.CreateTruncating(T.Log(eps * _05) / T.Log(ro1));//[SNR] p.300,(7)
				if (maxIters <= 0) maxIters = 5;
				tauk = new T[maxIters];
				int[] cheb = UtilsChebysh.chebyshParams(maxIters);
				for (int iter = 0; iter < maxIters; iter++)
				{
					T tk = -T.Cos((T.Pi * T.CreateTruncating(cheb[iter])) / T.CreateTruncating(2 * maxIters));//maxIters is n, k = 1,2,...,n
					tauk[iter] = tau0 / (T.One + ro0 * tk);//[SNR] p.299,(6)
				}
			}
		}

		void funcLapEqualStepsNoCheb(int i, int j) => un1[i, j] = (un0[i - 1, j] + un0[i + 1, j] + un0[i, j - 1] + un0[i, j + 1]) * _025;//fn == null
		void funcPoiEqualStepsNoCheb(int i, int j) => un1[i, j] = (un0[i - 1, j] + un0[i + 1, j] + un0[i, j - 1] + un0[i, j + 1] + fn[i, j] * stepX2) * _025;
		void funcLapEqualSteps(int i, int j) => un1[i, j] = un0[i, j] + tau1 * (UtilsOpLap.operatorLaplace(un0, i, j, _4) / stepX2);//fn == null
		void funcPoiEqualSteps(int i, int j) => un1[i, j] = un0[i, j] + tau1 * (UtilsOpLap.operatorLaplace(un0, i, j, _4) / stepX2 + fn[i, j]);//fn is NOT multiplied by step2(it is inside tau)
		void funcLap(int i, int j) => un1[i, j] = un0[i, j] + tau1 * ((un0[i - 1, j] - un0[i, j] * _2 + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] - un0[i, j] * _2 + un0[i, j + 1]) / stepY2);//fn == null
		void funcPoi(int i, int j) => un1[i, j] = un0[i, j] + tau1 * ((un0[i - 1, j] - un0[i, j] * _2 + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] - un0[i, j] * _2 + un0[i, j + 1]) / stepY2 + fn[i, j]);//fn is NOT multiplied by step2 

		//Explicit scheme,tau = h*h/(4*sigma),sigma==1
		public T doIteration(int iter)//O(h*h);simple iteration Jacobi method;simultaneous displacement method;Richardson method(if isChebysh)
		{
			T rc = default;
			tau1 = isChebysh ? tauk[iter] : tau;//used in actionCalculate
			if (isChebysh)
			{
				GridIterator.iterate(upper1, upper2, actionCalculate);//no need to calc delta - fixed number of iterations
				rc = eps + eps;
			}
			else rc = GridIterator.iterateForMaxWithEps<T>(upper1, upper2, actionCalculate, (i, j) => T.Abs(un0[i, j] - un1[i, j]), eps);

			UtilsSwap.swap(ref un0, ref un1);
			return rc;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			UtilsBorders.copyLeftRightValues(un0, un1);
			UtilsBorders.copyTopBottomValues(un0, un1);
		}

		public int maxIterations() { return maxIters; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			un0 = null;
			un1 = null;
			fn = null;
			tauk = null;
		}

		public override IterationsKind iterationsKind()
		{
			return isChebysh ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}
	}
}

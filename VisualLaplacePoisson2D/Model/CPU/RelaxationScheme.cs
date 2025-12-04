using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class RelaxationScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T[,] fn;
		readonly Func<int, int, T> func;
		readonly T stepX2, stepY2, eps, coef, rJacobi2, _025 = T.CreateTruncating(0.25);
		T omegaCoef, oneMinusOmega, omega;
		bool isChebysh;
		T _05 = T.CreateTruncating(0.5);
		T _2 = T.CreateTruncating(2);
		T _4 = T.CreateTruncating(4);

		public RelaxationScheme(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, bool isSeidel, bool isChebyshIn, T eps)
		{
			un0 = new T[cXSegments + 1, cYSegments + 1];
			this.eps = eps;
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			bool equalSteps = T.Abs(stepX - stepY) < T.Min(stepX, stepY) / T.CreateTruncating(100);//less than one percent
			T _2 = T.CreateTruncating(2.0);
			if (!equalSteps) coef = T.One / (_2 / stepX2 + _2 / stepY2);
			isChebysh = !isSeidel && isChebyshIn;

			T _cXSegments = T.CreateTruncating(cXSegments);
			T _cYSegments = T.CreateTruncating(cYSegments);
			if (!isChebysh)
			{
				T sinX = T.Sin(T.Pi / (_2 * _cXSegments));//[SNR] p.382(bottom)
				T sinY = T.Sin(T.Pi / (_2 * _cYSegments));//[SNR] p.382(bottom)
				T sumStep2 = stepX2 + stepY2;
				T lyambdaMin = _2 * stepY2 / sumStep2 * sinX * sinX + _2 * stepX2 / sumStep2 * sinY * sinY;//[SNR] p.382(bottom)
				omega = _2 / (T.One + T.Sqrt(lyambdaMin * (_2 - lyambdaMin)));//[SNR] p.379(14)
				if (equalSteps) omegaCoef = omega / _4;
				else omegaCoef = omega * coef;
				oneMinusOmega = T.One - omega;
			}
			else
			{
				omega = T.One;
				omegaCoef = omega / _4;
				oneMinusOmega = T.One - omega;
				omega = T.CreateTruncating(_05);//for omega = 1.0 / (1.0 - rJacobi2 / _2) be omega = 1.0 / (1.0 - rJacobi2 * omega / 4.0) on first iteration(iter==0)

				T rJacobi = (T.Cos(T.Pi / _cXSegments) + T.Cos(T.Pi / _cYSegments)) / _2;//NumericalRecipesinC,891,(19.5.24); deltaX == deltaY(equal steps in X & Y directions)
				rJacobi2 = rJacobi * rJacobi;
			}

			if (fKsi == null)
			{
				if (equalSteps)
				{
					if (isSeidel) func = funcLapEqualStepsSeidel;
					else func = funcLapEqualSteps;
				}
				else
				{
					if (isSeidel) func = funcLapSeidel;
					else func = funcLap;
				}
			}
			else
			{
				if (equalSteps)
				{
					if (isSeidel) func = funcPoiEqualStepsSeidel;
					else func = funcPoiEqualSteps;
				}
				else
				{
					if (isSeidel) func = funcPoiSeidel;
					else func = funcPoi;
				}
				fn = new T[cXSegments + 1, cYSegments + 1];//exterior points are not used
				if (equalSteps) GridIterator.iterate(cXSegments, cYSegments, (i, j) => fn[i, j] = stepX2 * fKsi(stepX * T.CreateTruncating(i), stepX * T.CreateTruncating(j)));
				else GridIterator.iterate(cXSegments, cYSegments, (i, j) => fn[i, j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}

		}

		T funcLapEqualStepsSeidel(int i, int j) => _025 * (un0[i - 1, j] + un0[i + 1, j] + un0[i, j - 1] + un0[i, j + 1]);//fn is null
		T funcPoiEqualStepsSeidel(int i, int j) => _025 * (un0[i - 1, j] + un0[i + 1, j] + un0[i, j - 1] + un0[i, j + 1] + fn[i, j]);//fn already multiplied by step^2

		T funcLapSeidel(int i, int j) => coef * ((un0[i - 1, j] + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] + un0[i, j + 1]) / stepY2);//fn is null
		T funcPoiSeidel(int i, int j) => coef * ((un0[i - 1, j] + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] + un0[i, j + 1]) / stepY2 + fn[i, j]);//Pakulina direchlet_num.pdf,p.9(14)

		T funcLapEqualSteps(int i, int j) => omegaCoef * (un0[i - 1, j] + un0[i + 1, j] + un0[i, j - 1] + un0[i, j + 1]) + oneMinusOmega * un0[i, j];//fn null
		T funcPoiEqualSteps(int i, int j) => omegaCoef * (un0[i - 1, j] + un0[i + 1, j] + un0[i, j - 1] + un0[i, j + 1] + fn[i, j]) + oneMinusOmega * un0[i, j];//fn already multiplied by step^2 

		T funcLap(int i, int j) => omegaCoef * ((un0[i - 1, j] + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] + un0[i, j + 1]) / stepY2) + oneMinusOmega * un0[i, j];//fn null
		T funcPoi(int i, int j) => omegaCoef * ((un0[i - 1, j] + un0[i + 1, j]) / stepX2 + (un0[i, j - 1] + un0[i, j + 1]) / stepY2 + fn[i, j]) + oneMinusOmega * un0[i, j];

		public T doIteration(int iter)//O(h*h)
		{
			if (isChebysh)
			{
				omega = T.One / (T.One - rJacobi2 * omega / _4);
				omegaCoef = omega / _4;
				oneMinusOmega = T.One - omega;
				//Trace.WriteLine(String.Format("omega={0}", omega));
			}
			return GridIterator.iterateRedBlackForMax(un0, func, eps);
		}

		public void initAfterBoundariesAndInitialIterationInited() {}

		public int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			un0 = null;
			fn = null;
		}
	}
}

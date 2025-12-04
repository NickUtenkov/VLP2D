using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class PTMSchemeBase<T> : Iterative2DScheme<T>, IScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		protected T ksi1, ksi2, tau, ksiDivider, tau1, eps, stepX2, stepY2;
		protected T[] tauk;//chebish
		protected readonly int maxIters;
		protected T[,] un1, fn;
		protected bool isChebysh;

		public PTMSchemeBase(int cXSegments, int cYSegments, T stepX, T stepY, bool isChebyshIn, T epsIn, Func<T, T, T> fKsi)
		{
			un0 = new T[cXSegments + 1, cYSegments + 1];
			un1 = new T[cXSegments + 1, cYSegments + 1];
			eps = epsIn;
			isChebysh = isChebyshIn;
			if (fKsi != null)
			{
				int upper1 = un0.GetUpperBound(0);
				int upper2 = un0.GetUpperBound(1);
				fn = new T[upper1 + 1, upper2 + 1];//exterior points are not used
				GridIterator.iterate(upper1, upper2, (i, j) => fn[i, j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}

			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;

			T _2 = T.CreateTruncating(2.0);
			T _4 = T.CreateTruncating(4.0);
			T piHalf = T.Pi / _2;
			T sinX = T.Sin(piHalf / T.CreateTruncating(cXSegments));
			T sinY = T.Sin(piHalf / T.CreateTruncating(cYSegments));
			T eigenMin = _4 * sinX * sinX / stepX2 + _4 * sinY * sinY / stepY2;//Pakulina direchlet_num.pdf,p.11(21)
			T eigenMax = _4 / stepX2 + _4 / stepY2;//Pakulina direchlet_num.pdf,p.11(21)
			T omega = _2 / T.Sqrt(eigenMin * eigenMax);//Pakulina direchlet_num.pdf,p.11(22)

			ksi1 = omega / stepX2;//Pakulina direchlet_num.pdf,p.11 between (27) & (28)
			ksi2 = omega / stepY2;//Pakulina direchlet_num.pdf,p.11 between (27) & (28)
			ksiDivider = T.One / (T.One + ksi1 + ksi2);// Pakulina direchlet_num.pdf,p.11 (28)

			T nuSqrt = T.Sqrt(eigenMin / eigenMax);
			T gamma1 = eigenMin / (_2 + _2 * nuSqrt);//Pakulina direchlet_num.pdf,p.11 (22)
			T gamma2 = eigenMin / (_4 * nuSqrt);//Pakulina direchlet_num.pdf,p.11 (22)
			tau = _2 / (gamma1 + gamma2);//Pakulina direchlet_num.pdf,p.11(22)

			if (isChebysh)
			{
				maxIters = (int)double.Ceiling(double.CreateTruncating(T.Log(_2 / eps) / (_2 * T.Sqrt(_2) * T.Sqrt(nuSqrt))));//Pakulina direchlet_num.pdf,p.13
				tauk = new T[maxIters];
				int[] cheb = UtilsChebysh.chebyshParams(maxIters);
				for (int iter = 0; iter < maxIters; iter++)
				{
					T tk = -T.Cos((T.Pi * T.CreateTruncating(cheb[iter])) / T.CreateTruncating(2 * maxIters));//maxIters is n, k = 1,2,...,n
					tauk[iter] = _2 / (gamma1 + gamma2 + (gamma2 - gamma1) * tk);//Pakulina direchlet_num.pdf,p.12(33)
				}
			}
		}

		virtual public T doIteration(int iter)
		{
			return T.Zero;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			UtilsBorders.copyLeftRightValues(un0, un1);
			UtilsBorders.copyTopBottomValues(un0, un1);
		}

		public override IterationsKind iterationsKind()
		{
			return isChebysh ? IterationsKind.knownInAdvance : IterationsKind.unknown;
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
	}
}

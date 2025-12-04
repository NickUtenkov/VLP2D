using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using VLP2D.Common;

namespace VLP2D.Model
{
	class ProgonkaScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>
	{
		protected T stepX2, stepY2, dt, eps;
		protected T[,] un1, fn;
		T[,] unm;
		protected T[] alphaX, alphaY;//beta is placed to un1
		protected int kX, kY;
		ParallelOptions optionsParallel;
		protected int cXSegments, cYSegments;
		protected Func<T[,], int, int, int, T> rhsX, rhsY;//Right Hand Sides
		protected Func<int, int, T> funcX, funcY;
		protected T[,] srcX, dstX, srcY, dstY;//source & destination array
		protected bool bProgonkaFixedIters;
		protected Action<int> calculateIterationAlpha = null;
		protected AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		protected T _2 = T.CreateTruncating(2);

		public ProgonkaScheme(int cXSegments1, int cYSegments1, T stepX, T stepY, T epsIn, Func<T, T, T> fKsi, ParallelOptions optionsParallelIn)
		{
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			cXSegments = cXSegments1;
			cYSegments = cYSegments1;

			un0 = new T[cXSegments + 1, cYSegments + 1];
			un1 = new T[cXSegments + 1, cYSegments + 1];
			unm = new T[cXSegments + 1, cYSegments + 1];

			optionsParallel = optionsParallelIn;

			alphaX = new T[cXSegments];
			alphaY = new T[cYSegments];

			eps = epsIn;
			if (fKsi != null)
			{
				fn = new T[cXSegments + 1, cYSegments + 1];//exterior points are not used
				GridIterator.iterate(cXSegments, cYSegments, (i, j) => fn[i, j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
			}

			calculateOptimalTimeStep(stepX, stepY);
			setSrcDst();
			dstX = unm;
			srcY = unm;
		}

		void calculateOptimalTimeStep(T stepX, T stepY)
		{
			T a = stepX * T.CreateTruncating(cXSegments);
			T b = stepY * T.CreateTruncating(cYSegments);
			dt = T.Sqrt(stepX2 + stepY2) * T.Pow(T.One / (a * a) + T.One / (b * b), T.CreateTruncating(-0.5)) / T.Pi;//Kalitkin p.406 (19)
		}

		protected void calcAlpha(T bx, T by)
		{
			kX = αCC.upperBound(bx, cXSegments - 1);
			alphaX[0] = T.Zero;
			for (int i = 1; i <= kX; i++) alphaX[i] = T.One / (bx - alphaX[i - 1]);

			kY = αCC.upperBound(by, cYSegments - 1);
			alphaY[0] = T.Zero;
			for (int i = 1; i <= kY; i++) alphaY[i] = T.One / (by - alphaY[i - 1]);
		}

		public T doIteration(int iter)
		{
			calculateIterationAlpha?.Invoke(iter);

			Parallel.For(1, cYSegments, optionsParallel, j => progonkaX(srcX, dstX, j, iter));
			Parallel.For(1, cXSegments, optionsParallel, i => progonkaY(srcY, dstY, i, iter));

			T rc;
			if (bProgonkaFixedIters) rc = T.One;
			else rc = (GridIterator.iterateUntilCondition(un0.GetUpperBound(0), un0.GetUpperBound(1), (i, j) => T.Abs(un0[i, j] - un1[i, j]) > eps) ? T.One : T.Zero);

			UtilsSwap.swap(ref un0, ref un1);
			setSrcDst();

			return rc;
		}

		void setSrcDst()
		{
			srcX = un0;
			dstY = un1;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			UtilsBorders.copyLeftRightValues(un0, un1);
			UtilsBorders.copyTopBottomValues(un0, un1);

			UtilsBorders.copyLeftRightValues(un0, unm);
			UtilsBorders.copyTopBottomValues(un0, unm);
		}

		public virtual int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public virtual void cleanup()
		{
			un0 = null;
			un1 = null;
			unm = null;
			fn = null;
			alphaX = null;
			alphaY = null;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		protected T operatorLxx(T[,] u, int i, int j)
		{//using without dividing by step2(not needed in some cases)
			return u[i - 1, j] + u[i + 1, j] - u[i, j] * _2;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		protected T operatorLyy(T[,] u, int i, int j)
		{//using without dividing by step2(not needed in some cases)
			return u[i, j - 1] + u[i, j + 1] - u[i, j] * _2;
		}

		void progonkaX(T[,] src, T[,] dst, int j, int iter)//original name progonx
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			int ind(int idx) => (idx < kX) ? idx : kX;
			//dst[0, j] = src[0, j];//already assigned in init boundaries,1st kind boundary condition
			for (int i = 1; i < cXSegments; i++) dst[i, j] = alphaX[ind(i)] * (rhsX(src, i, j, iter) + dst[i - 1, j]);
			for (int i = cXSegments - 1; i > 0; i--) dst[i, j] += alphaX[ind(i)] * dst[i + 1, j];
		}

		void progonkaY(T[,] src, T[,] dst, int i, int iter)
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			int ind(int idx) => (idx < kY) ? idx : kY;
			//dst[i, 0] = src[i, 0];//already assigned in init boundaries
			for (int j = 1; j < cYSegments; j++) dst[i, j] = alphaY[ind(j)] * (rhsY(src, i, j, iter) + dst[i, j - 1]);
			for (int j = cYSegments - 1; j > 0; j--) dst[i, j] += alphaY[ind(j)] * dst[i, j + 1];
		}
	}
}

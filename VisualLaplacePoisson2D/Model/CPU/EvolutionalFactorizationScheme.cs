using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using VLP2D.Common;

namespace VLP2D.Model
{
	//https://bitbucket.org/alexander_belov/sufarec/downloads/
	//Программы SuFaReC для сверхбыстрого расчета эллиптических уравнений в прямоугольной области.pdf
	//Белов Калиткин Эволюционная факторизация и сверхбыстрый счет на установление.pdf
	//Калиткин Н. Н. Улучшенная факторизация параболических схем.pdf
	//https://github.com/ABelov91 - abscent SuFaReC
	class EvolutionalFactorizationScheme<T> : Iterative2DScheme<T>, IScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>//, IFloatingPoint<T>
	{
		T eps, hx2, hy2, twoDivTau, xMax, yMax;
		T[,] un1, unm, fn;
		T[] alphaX, alphaY, τ;//beta is placed to dst
		int kX = int.MaxValue, kY = int.MaxValue;
		int cXSegments, cYSegments;
		Func<T[,], int, int, T> rhsX, rhsY;//Right Hand Sides
		T[,] srcX, dstY;//source & destination array
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		static T _2 = T.CreateTruncating(2), _4 = T.CreateTruncating(4);

		public EvolutionalFactorizationScheme(RectangleData<T> rectData, T eps, Func<T, T, T> fKsi) :
			base(rectData.xMin, rectData.yMin, rectData.stepX, rectData.stepY)
		{
			this.cXSegments = rectData.cXSegments;
			this.cYSegments = rectData.cYSegments;
			this.eps = eps;
			xMax = rectData.xMax;
			yMax = rectData.yMax;

			T stepX2 = stepX * stepX;
			T stepY2 = stepY * stepY;
			T ratioOfSquaresOfSteps = stepX2 / stepY2;

			un0 = new T[cXSegments + 1, cYSegments + 1];
			un1 = new T[cXSegments + 1, cYSegments + 1];
			unm = new T[cXSegments + 1, cYSegments + 1];

			alphaX = new T[cXSegments];
			alphaY = new T[cYSegments];

			bool isPoisson = fKsi != null;
			if (isPoisson)
			{
				fn = new T[cXSegments + 1, cYSegments + 1];//exterior points are not used
				GridIterator.iterate(cXSegments, cYSegments, (i, j) => fn[i, j] = fKsi(xMin + stepX * T.CreateTruncating(i), yMin + stepY * T.CreateTruncating(j)));
			}
			Func<int, int, T> right_hand = (i, j) => isPoisson ? fn[i, j] : T.Zero;

			Func<T[,], int, int, T> F1 = (src, i, j) => operatorLxx(src, i, j);//operator Λₓ
			Func<T[,], int, int, T> F2 = (src, i, j) => operatorLyy(src, i, j) * ratioOfSquaresOfSteps;//operator Λᵧ
			rhsX = (src, i, j) => (right_hand(i, j) * stepX2 + F1(src, i, j) + F2(src, i, j)) * twoDivTau;
			rhsY = (src, i, j) => src[i, j] * twoDivTau * hy2;

			hx2 = stepX2 / _4;//== (stepX/2)^2
			hy2 = stepY2 / _4;//== (stepY/2)^2

			τ = UtilsLT<T>.calculateTau(eps, hx2, hy2, xMax - xMin, yMax - yMin);

			setSrcDst();
		}	

		public T doIteration(int iter)
		{
			evolutionalFactorization(τ[iter]);

			UtilsSwap.swap(ref un0, ref un1);
			setSrcDst();

			return T.One;
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
		}

		public virtual int maxIterations() { return τ.Length; }
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

		public override IterationsKind iterationsKind() => IterationsKind.knownInAdvance;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		protected T operatorLxx(T[,] u, int i, int j)
		{//using without dividing by step2(not needed in some cases)
			return u[i - 1, j] - u[i, j] * _2 + u[i + 1, j];
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		protected T operatorLyy(T[,] u, int i, int j)
		{//using without dividing by step2(not needed in some cases)
			return u[i, j - 1] - u[i, j] * _2 + u[i, j + 1];
		}

		void evolutionalFactorization(T τ)
		{
			twoDivTau = _2 / τ;

			calcAlpha(_2 + twoDivTau * hx2, _2 + twoDivTau * hy2);
			Parallel.For(1, cYSegments, GridIterator.optionsParallel, k => progonkaX(srcX, unm, k));
			Parallel.For(1, cXSegments, GridIterator.optionsParallel, n => progonkaY(unm, dstY, srcX, n, τ));
		}

		protected void calcAlpha(T bx, T by)
		{
			Action<T, T[], int> calc = (diag, alpha, bound) =>
			{
				alpha[0] = T.Zero;
				for (int i = 1; i <= bound; i++) alpha[i] = T.One / (diag - alpha[i - 1]);
			};

			kX = αCC.upperBound(bx, cXSegments - 1);
			kY = αCC.upperBound(by, cYSegments - 1);

			Action[] actions = [() => calc(bx, alphaX, kX), () => calc(by, alphaY, kY)];
			Parallel.For(0, 2, GridIterator.optionsParallel, j => actions[j].Invoke());
		}

		void progonkaX(T[,] src, T[,] dst, int j)
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			int ind(int idx) => (idx < kX) ? idx : kX;

			for (int i = 1; i < cXSegments; i++) dst[i, j] = alphaX[ind(i)] * (rhsX(src, i, j) + (i != 1 ? dst[i - 1, j] : T.Zero));
			for (int i = cXSegments - 1; i > 0; i--) dst[i, j] += alphaX[ind(i)] * (i != cXSegments - 1 ? dst[i + 1, j] : T.Zero);
		}

		void progonkaY(T[,] src, T[,] dst, T[,] srcX, int i, T τ)
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			int ind(int idx) => (idx < kY) ? idx : kY;

			for (int j = 1; j < cYSegments; j++) dst[i, j] = alphaY[ind(j)] * (rhsY(src, i, j) + (j != 1 ? dst[i, j - 1] : T.Zero));
			for (int j = cYSegments - 1; j > 0; j--) dst[i, j] += alphaY[ind(j)] * (j != cYSegments - 1 ? dst[i, j + 1] : T.Zero);

			//post proccess progonka
			for (int j = 1; j <= cYSegments - 1; j++) dst[i, j] = (srcX[i, j] + τ * dst[i, j]);
		}
	}
}

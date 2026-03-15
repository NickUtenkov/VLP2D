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
		T eps, hx2, hy2;
		T[,] un1, unm, fn;
		T[] alphaX, alphaY;//beta is placed to dst
		int kX = int.MaxValue, kY = int.MaxValue;
		int cXSegments, cYSegments;
		T xMax, yMax;
		Func<T[,], int, int, T> rhsX, rhsY;//Right Hand Sides
		T[,] srcX, dstY;//source & destination array
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		static T _05 = T.CreateTruncating(0.5);
		static T _2 = T.CreateTruncating(2);
		static T _4 = T.CreateTruncating(4);
		T[] τ;
		T  twoDivTau;

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

			τ = calculateTau();

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

		T[] calculateTau()
		{
			T conditioning;
			T[] lambda_max, lambda_min;
			(lambda_max, lambda_min, conditioning) = spectrumEstimates();
			T epsilon_background = T.CreateTruncating(UtilsEps.epsilon<T>()) * conditioning;//can use UtilsEps.epsilonBackground<T>()
			T eps_grid_sys = eps;
			T epsilon = T.Max(epsilon_background, eps_grid_sys);
			int[] sAll = numberOfSteps(double.CreateTruncating(conditioning), double.CreateTruncating(epsilon));
			List<T> listτ = new List<T>();
			for (int i = 0; i < sAll.Length; i++)
			{
				int S = sAll[i];
				T[] tau_lt = logarithmicSet(lambda_max, lambda_min, S);
				if (i == 0) listτ.AddRange(tau_lt);
				else for (int s = 1; s <= S / 2; s++) listτ.Add(tau_lt[2 * s - 1]);
			}

			return listτ.ToArray();
		}

		(T[], T[], T) spectrumEstimates()
		{//Constructs estimates for spectrum boundaries of the grid system
			T est_x = _4 / hx2;
			T est_y = _4 / hy2;
			T[] lambda_max = [est_x, est_y];
			T lx = xMax - xMin;
			T ly = yMax - yMin;
			T pilx = T.Pi / lx;
			T pily = T.Pi / ly;
			T[] lambda_min = [pilx * pilx, pily * pily];
			T conditioning = (lambda_max[0] + lambda_max[1]) / (lambda_min[0] + lambda_min[1]);
			return (lambda_max, lambda_min, conditioning);
		}

		int[] numberOfSteps(double relation, double epsilon)
		{// Calculates the numbers of steps in nested logarithmic grids via a priori convergence estimate
			int count = 10;
			double[] S_temp = new double[count];
			int I = 0;
			S_temp[0] = double.Ceiling(-(4 / (Math.PI * Math.PI + 2 * Math.PI)) * Math.Log(relation) * Math.Log(epsilon));
			while (S_temp[I] >= 2)
			{
				S_temp[I + 1] = S_temp[I] / 2;
				I = I + 1;
				if (I == count) break;
			}

			int[] sAll = new int[I];
			double ceil = double.Ceiling(S_temp[I]);
			for (int m = 0; m < I; m++) sAll[m] = (int)(ceil * Math.Pow(2, m));

			return sAll;
		}

		T[] logarithmicSet(T[] lambda_max, T[] lambda_min, int S)
		{// Constructs linear-trigonometric set in logarithmic scale with given number of steps
			T τMin = _2 / (lambda_max[0] + lambda_max[1]);
			T τMax = _2 / (lambda_min[0] + lambda_min[1]);
			T center = _05 * T.Log(τMin * τMax);
			T width = _05 * T.Log(τMax / τMin);
			T piPlus2 = T.Pi + _2;
			T _2Div = _2 / piPlus2;
			T piDiv = T.Pi / piPlus2;

			T[] τ = new T[S + 1];
			T _S = T.CreateTruncating(S);
			for (int s = 0; s <= S; s++)
			{
				T θ = T.CreateTruncating(s) / _S;
				T lt = _2Div * T.Cos(θ * T.Pi - T.Pi) + piDiv * (_2 * θ - T.One);
				τ[s] = T.Pow(T.E, center + width * lt);
			}

			return τ;
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

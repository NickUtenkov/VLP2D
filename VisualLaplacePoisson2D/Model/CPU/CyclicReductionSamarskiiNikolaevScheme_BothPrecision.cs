
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class CyclicReductionSamarskiiNikolaevScheme_BothPrecision<T> : CyclicReductionScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T[][] ac,vv;//accumulator
		T[][] p;
		T stepX2, stepY2, derivativeCoeff;
		Func<T[][], int, int, T> operatorB;
		bool morePrecision;
		Coeffs coeffs;
		T _1 = T.One;
		T _2 = T.CreateTruncating(2.0);
		T _12 =T.CreateTruncating(12.0);
		T _16 = T.CreateTruncating(16.0);
		T _30 = T.CreateTruncating(30.0);

		public CyclicReductionSamarskiiNikolaevScheme_BothPrecision(int cXSegments, int cYSegments, T stepXIn, T stepYIn, bool morePrecision, int cCores, Func<T, T, T> fKsi, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap0, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, stepXIn, stepYIn, cCores, fKsi, lstBitmap0, fCreateBitmap0, reportProgressIn)
		{
			//originally(in math description - [SNR] p.138) p is (N1 - 1)*(N2 - 1) size, but share the same memory with un(which is solution)
			//here it is interior part of un, which is (N1 + 1)*(N2 + 1) size
			//so it is used with shifted by 1 indeces
			p = un;

			ac = new T[cCores][];//use less memory - cCores*(N2 - 1) instead of (N1 - 1)*(N2 - 1)
			vv = new T[cCores][];//use less memory - cCores*(N2 - 1) instead of (N1 - 1)*(N2 - 1)
			for (int i = 0; i < cCores; i++)
			{
				ac[i] = new T[N2];//extra zero index not used
				vv[i] = new T[N2];//extra zero index not used
			}

			alfa = new T[cCores][];//redo ?! differs from other methods
			for (int i = 0; i < cCores; i++) alfa[i] = new T[N2];
			alfaUpperBound = progonkaUpperBound;

			stepX2 = stepXIn * stepXIn;
			stepY2 = stepYIn * stepYIn;
			this.morePrecision = morePrecision;
			derivativeCoeff = morePrecision ? (stepX2 + stepY2) / _12 : T.Zero;
			if (morePrecision) operatorB = operatorBMorePrecision;
			else operatorB = operatorBOrdinaryPrecision;

			progressSteps = (n - 1) + n;//(n - 1) - direct steps, n - reverse steps

			coeffs = new Coeffs(n, morePrecision, bCoef, stepX2, stepY2, cosKL);
		}

		override public T doIteration(int iter)
		{//Q = 9.5*N2*N1*Log(N1,2) - 8*N2*N1;N2 can be == N1
		 // p_j(0)  = F_j, [SNR] p.138 1) - not (1)
			int cCores = optionsParallel.MaxDegreeOfParallelism;
			initRigthHandSide(p);
			transferBottomTopToInterior(p);

			forwardWay();
			if (areIterationsCanceled()) return T.Zero;

			reverseWay();
			if (areIterationsCanceled()) return T.Zero;

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		void forwardWay()
		{
			T _05 = T.CreateTruncating(0.5);
			int cCores = optionsParallel.MaxDegreeOfParallelism;
			for (int k = 1; k <= n - 1; k++)
			{
				int m = 1 << (k - 1);
				int t = m << 1;

				int count1 = N1 / t - 1;//from t - 1 to N1 - t - 1 step t (+ 1 for count)
				int loopCount = Math.Min(count1, cCores);
				Parallel.For(0, loopCount, optionsParallel, core =>
				{
					for (int idx = core; idx < count1; idx += cCores)
					{
						int j = t * (idx + 1);
						for (int l = 1; l <= m; l++) // l = 1..2^{k-1}
						{
							DiagElemCoefBCoefAlfa data = coeffs.getCoeffs(l, k);
							calcAlpha(data.diagElem, alfa[core]);
							progonka((i) => data.coefAlfa * (operatorB(p, j - m, i) + operatorB(p, j + m, i)), data.coefB, alfa[core], vv[core]);//[SNR] p.148, replacement p.138 (21-22)
							for (int i = 1; i < N2; i++) p[j][i] += vv[core][i];//[SNR] p.138, (23)(also see below)
						}
						for (int i = 1; i < N2; i++) p[j][i] *= _05;//[SNR] p.138, (23)
					}
				});

				showProgress();
			}
		}

		void reverseWay()
		{
			int cCores = optionsParallel.MaxDegreeOfParallelism;
			for (int k = n; k >= 1; k--)
			{
				int m = 1 << (k - 1);
				int t = m << 1;

				int count2 = N1 / t;//from m - 1 to N1 - m - 1 step t (+ 1 for count)
				int loopCount = Math.Min(count2, cCores);
				Parallel.For(0, loopCount, optionsParallel, core =>
				{
					for (int idx = core; idx < count2; idx += cCores)
					{
						int j = m + idx * t;
						Array.Clear(ac[core], 1, N2 - 1);
						for (int l = 1; l <= m; l++) // l = 1..2^{k-1}
						{
							DiagElemCoefBCoefAlfa data = coeffs.getCoeffs(l, k);
							calcAlpha(data.diagElem, alfa[core]);
							progonka((i) => p[j][i] + data.coefAlfa * (operatorB(un, j - m, i) + operatorB(un, j + m, i)), data.coefB, alfa[core], vv[core]);//[SNR] p.148, replacement for (24) & p.138 (25)
							for (int i = 1; i < N2; i++) ac[core][i] += vv[core][i];//[SNR] p.138, (26)(also see below)
						}
						for (int i = 1; i < N2; i++) un[j][i] = ac[core][i];//[SNR] p.138, (26)
						if (unShow != null) for (int i = 1; i < N2; i++) unShow[j][i] = float.CreateTruncating(un[j][i]);
					}
				});

				if (k != 1 && unShow != null)
				{//if k == 1, then picture will be the same as final(which is added in external code)
					UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (i, j) => unShow[i][j]), fCreateBitmap);
				}
				showProgress();
			}
		}

		new void transferBottomTopToInterior(T[][] fj)//not using base class func because of use of derivativeCoeff
		{
			T mult = T.One / bCoef;

			Parallel.For(1, fj.GetUpperBound(0), (j) =>
			{//additional init of F_j; fi overline,[SNR] p.129 middle
				fj[j][1] += mult * (un[j][0] + derivativeCoeff * operatorDerivativeSecondX(un, j, 0));
				fj[j][N2 - 1] += mult * (un[j][N2] + derivativeCoeff * operatorDerivativeSecondX(un, j, N2));
			});
			//Utils.printArray(un, "un after transfer", "{0,9:0.00E+00}", 0);

			//[MethodImpl(MethodImplOptions.AggressiveInlining)]
			T operatorDerivativeSecondX(T[][] u, int j, int i)
			{//j - vector number,i - its coordinates
				return (u[j - 1][i] - _2 * u[j][i] + u[j + 1][i]) / stepX2;
			}
		}

		void calcAlpha(T diagElem, T[] alfa)
		{
			alfa[0] = T.One / diagElem;//[SNR] p.145 bottom
			for (int i = 1; i <= alfaUpperBound; i++) alfa[i] = T.One / (diagElem - alfa[i - 1]);//[SNR] p.145 bottom
		}

		void progonka(Func<int, T> rhs, T coeffB, T[] alfa, T[] res)
		{//[SNR] p.145 bottom
			//beta array is not used - it is placed to res
			res[1] = coeffB * rhs(1) * alfa[0];
			for (int i = 1; i <= progonkaUpperBound; i++) res[i + 1] = (coeffB * rhs(i + 1) + res[i]) * alfa[i];

			for (int i = progonkaUpperBound - 1; i >= 0; i--) res[i + 1] += alfa[i] * res[i + 2];
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		T operatorBMorePrecision(T[][] y,int j0,int i0)
		{//j - vector number,i - its coordinates
			return y[j0][i0] + derivativeCoeff * operatorDerivativeSecondY(y, j0, i0);//[SNR] p.147 between (6) & (7)

			//[MethodImpl(MethodImplOptions.AggressiveInlining)]
			T operatorDerivativeSecondY(T[][] u, int j, int i)
			{//j - vector number,i - its coordinates
#if !Oh4
				if (i == 1) return (T.Zero - _2 * u[j][i] + u[j][i + 1]) / stepY2;//assume u[j][0] == 0
				if (i == N2 - 1) return (u[j][i - 1] - _2 * u[j][i] + T.Zero) / stepY2;//assume u[j][N2] == 0
				return (u[j][i - 1] - _2 * u[j][i] + u[j][i + 1]) / stepY2;
#else
				if (i == 1) return (-_1 * T.Zero + _16 * T.Zero - _30 * u[j][i] + _16 * u[j][i + 1] - _1 * u[j][i + 2]) / (_12 * stepY2);//assume u[j][0] == 0, u[j][-1] == 0
				if (i == 2) return (-_1 * T.Zero + _16 * u[j][i - 1] - _30 * u[j][i] + _16 * u[j][i + 1] - _1 * u[j][i + 2]) / (_12 * stepY2);//assume u[j][0] == 0, u[j][-1] == 0
				if (i == N2 - 1) return (-_1 * u[j][i - 2] + _16 * u[j][i - 1] - _30 * u[j][i] + _16 * T.Zero - _1 * T.Zero) / (_12 * stepY2);//assume u[j][N2] == 0, u[j][N2+1] == 0
				if (i == N2 - 2) return (-_1 * u[j][i - 2] + _16 * u[j][i - 1] - _30 * u[j][i] + _16 * u[j][i + 1] - _1 * T.Zero) / (_12 * stepY2);//assume u[j][N2] == 0, u[j][N2+1] == 0
				return (-_1 * u[j][i - 2] + _16 * u[j][i - 1] - _30 * u[j][i] + _16 * u[j][i + 1] - _1 * u[j][i + 2]) / (_12 * stepY2);
#endif
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		T operatorBOrdinaryPrecision(T[][] y, int j, int i) => y[j][i];//j - vector number,i - its coordinates

		struct DiagElemCoefBCoefAlfa
		{
			public T coefAlfa, coefB;
			public T diagElem;
			public DiagElemCoefBCoefAlfa(T diagElem, T coefB, T coefAlfa)
			{
				this.diagElem = diagElem;//[SNR] p.145 (4), p.149 (10)
				this.coefB = coefB;//[SNR] p.145 (4), p.149 (10)
				this.coefAlfa = coefAlfa;//[SNR] p.136 (20), p.138 (22)
			}
		}

		class Coeffs
		{
			Dictionary<int, DiagElemCoefBCoefAlfa> dictDiagElemCoefBCoefAlfa;
			bool morePrecision;
			int n, dictKeyShift;
			T stepX2, stepY2, bCoef;
			Func<int, int, T> cosKL;
			T _2 = T.CreateTruncating(2.0);
			T _5 = T.CreateTruncating(5.0);
			T _6 = T.CreateTruncating(6.0);

			public Coeffs(int n, bool morePrecision, T bCoef, T stepX2, T stepY2, Func<int, int, T> cosKL)
			{
				this.n = n;
				this.morePrecision = morePrecision;
				this.bCoef = bCoef;
				this.stepX2 = stepX2;
				this.stepY2 = stepY2;
				this.cosKL = cosKL;

				dictKeyShift = (int)Math.Ceiling(Math.Log(n, 2));

				dictDiagElemCoefBCoefAlfa = new Dictionary<int, DiagElemCoefBCoefAlfa>();
				fillDictDiagElemCoefBCoefAlfa(morePrecision);
			}

			public DiagElemCoefBCoefAlfa getCoeffs(int l, int k) => dictDiagElemCoefBCoefAlfa[dictKey(l, k)];

			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			int dictKey(int l, int k)
			{
				return (l << dictKeyShift) + k;
			}

			void fillDictDiagElemCoefBCoefAlfa(bool morePrecision)
			{
				for (int k = 1; k <= n; k++)
				{
					int m = 1 << (k - 1);
					T _2ᵏ = T.CreateTruncating(m << 1);
					T _m = T.CreateTruncating(m);
					for (int l = 1; l <= m; l++) // l = 1..2^{k-1}
					{
						int key = dictKey(l, k);
						if (!dictDiagElemCoefBCoefAlfa.ContainsKey(key))
						{
							T cos = cosKL(k, l);
							T b = morePrecision ? coeffB(cos) : bCoef;
							dictDiagElemCoefBCoefAlfa[key] = new DiagElemCoefBCoefAlfa(diagonalElement(b, cos), b, alphaCoeff(l, _2ᵏ, _m));
						}
					}
				}

				T diagonalElement(T b, T cos)
				{
					return _2 * (T.One + b * (T.One - cos));//[SNR] p.149, (10);see also p.145, (4)
				}

				T coeffB(T cos)
				{
					T rc = _6 * stepY2 / (_5 * stepX2 - stepY2 + (stepY2 + stepX2) * cos);//[SNR] p.149, (10);see also p.145, (4)
					return rc;
				}

				T alphaCoeff(int l, T _2ᵏ, T m)
				{
					T sin = T.Sin(T.Pi * T.CreateTruncating(2 * l - 1) / _2ᵏ) / m;
					if (l % 2 == 0) sin = -sin;
					return sin;//[SNR] p136 (20)
				}
			}
		}
	}
}

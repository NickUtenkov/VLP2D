#define ExtraThreads

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class CyclicReductionSamarskiiNikolaevScheme<T> : CyclicReductionScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T[][] p;//==un
		T[][] ac;
		T[][] vv;
		T[] alphaCoeffs;
		T _05 = T.CreateTruncating(0.5);

		public CyclicReductionSamarskiiNikolaevScheme(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsi, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, stepXIn, stepYIn, cCores, fKsi, lstBitmap0, fCreateBitmap, reportProgressIn)
		{
			//originally(in math description - [SNR] p.138) p is (N1 - 1)*(N2 - 1) size, but share the same memory with un(which is solution)
			//here it is interior part of un, which is (N1 + 1)*(N2 + 1) size
			//so it is used with shifted by 1 indeces
			p = un;
#if ExtraThreads
			int arSize = cCores > 1 ? (cCores - 1) * cCores : 1;
#else
			int arSize = cCores;
#endif
			ac = new T[arSize][];
			vv = new T[arSize][];
			for (int i = 0; i < arSize; i++)
			{
				ac[i] = new T[N2];//extra zero index not used
				vv[i] = new T[N2];//extra zero index not used
			}

			alphaCoeffs = new T[N1 - 1];
			alfa = new T[N1 - 1][];

			progressSteps = (n - 1) + n;//(n - 1) - direct steps, n - reverse steps
		}

		override public T doIteration(int iter)
		{//Q = 9.5*N2*N1*Log(N1,2) - 8*N2*N1;N2 can be == N1
		 // p_j(0)  = F_j, [SNR] p.138 1) - not (1)
			initElapsedList();

			float elapsed = getExecutedSeconds(() => { fillAlphaCoeffs(); fillAlphaArrays(); initRigthHandSide(p); transferBottomTopToInterior(p); });
			listElapsedAdd("RHS, transfer", elapsed);

			elapsed = getExecutedSeconds(() => forwardWay());
			listElapsedAdd("forwardWay", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(() => reverseWay());
			listElapsedAdd("reverseWay", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		void forwardWay()
		{
			for (int k = 1; k <= n - 1; k++)
			{
				int m = 1 << (k - 1);//2ᵏ⁻¹
				int _2ᵏ = m << 1;
				int allVectors = N1 / _2ᵏ - 1;//1·2ᵏ, 2·2ᵏ, 3·2ᵏ, ..., (N1 - 2ᵏ); == 1,2,3,...,(N1 / 2ᵏ - 1) · 2ᵏ, [SNR] p.138, (21)
				int cCores0 = int.Min(optionsParallel.MaxDegreeOfParallelism, allVectors);
				Parallel.For(0, cCores0, optionsParallel, (core0) =>
				{
					for (int idx = core0 + 0; idx < allVectors; idx += cCores0)
					{
						int j = _2ᵏ * (idx + 1);
#if ExtraThreads
						directOrReverse(true, allVectors, core0, k, m, j);
#else
						for (int l = 1; l <= m; l++) // l = 1..2^{k-1}
						{
							int idxAlfa = m - 2 + l;
							T coef = alphaCoeffs[idxAlfa];
							Func<int, T> rhs = (i) => coef * (p[j - m][i] + p[j + m][i]);
							progonka(rhs, alfa[idxAlfa], vv[core0]);//[SNR] p.138 (21) & (22)
							for (int i = 1; i < N2; i++) p[j][i] += vv[core0][i];//[SNR] p.138, (23)(also see below)
						}
#endif
						for (int i = 1; i < N2; i++) p[j][i] *= _05;//[SNR] p.138, (23)
					}
				});

				showProgress();
				if (areIterationsCanceled()) return;
			}
		}

		void reverseWay()
		{
			for (int k = n; k >= 1; k--)// reverse steps
			{
				int m = 1 << (k - 1);//2ᵏ⁻¹
				int _2ᵏ = m << 1;
				int allVectors = N1 / _2ᵏ;//1·2ᵏ⁻¹, 3·2ᵏ⁻¹, 5·2ᵏ⁻¹, ..., (N1 - 2ᵏ⁻¹); == 1,3,5,...,(N1 / 2ᵏ⁻¹ - 1) · 2ᵏ⁻¹, [SNR] p.138, (24)
				int cCores0 = int.Min(optionsParallel.MaxDegreeOfParallelism, allVectors);
				Parallel.For(0, cCores0, optionsParallel, (core0) =>
				{
					for (int idx = core0 + 0; idx < allVectors; idx += cCores0)
					{
						int j = m + _2ᵏ * idx;
#if ExtraThreads
						directOrReverse(false, allVectors, core0, k, m, j);
#else
						Array.Clear(ac[core0], 1, N2 - 1);
						for (int l = 1; l <= m; l++) // l = 1..2^{k-1}
						{
							int idxAlfa = m - 2 + l;
							T coef = alphaCoeffs[idxAlfa];
							Func<int, T> rhs = (i) => p[j][i] + coef * (un[j - m][i] + un[j + m][i]);
							progonka(rhs, alfa[idxAlfa], vv[core0]);//[SNR] p.138 (24) & (25)
							for (int i = 1; i < N2; i++) ac[core0][i] += vv[core0][i];//[SNR] p.138, (26)(also see below)
						}
						for (int i = 1; i < N2; i++) un[j][i] = ac[core0][i];//[SNR] p.138, (26)
#endif
						if (unShow != null) for (int i = 1; i < N2; i++) unShow[j][i] = float.CreateTruncating(un[j][i]);
					}
				});

				if (k != 1 && unShow != null)
				{//if k == 1, then picture will be the same as final(which is added in external code)
					UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (i, j) => unShow[i][j]), fCreateBitmap);
				}
				showProgress();
				if (areIterationsCanceled()) return;
			}
		}

#if ExtraThreads
		void directOrReverse(bool bDirect, int allVectors, int core0, int k, int m, int j)
		{
			int cCores = allVectors < optionsParallel.MaxDegreeOfParallelism ? optionsParallel.MaxDegreeOfParallelism : 1;
			int coreOffset = core0 * cCores;
			Parallel.For(0, cCores, optionsParallel, (core) =>
			{
				int idxCore = coreOffset + core;
				Array.Clear(ac[idxCore], 1, N2 - 1);
				for (int l = core + 1; l <= m; l += cCores)
				{
					int idxAlfa = m - 2 + l;
					T coef = alphaCoeffs[idxAlfa];
					Func<int, T> rhs = (i) => bDirect ? coef * (p[j - m][i] + p[j + m][i]) : p[j][i] + coef * (un[j - m][i] + un[j + m][i]);
					progonka(rhs, alfa[idxAlfa], vv[idxCore]);//[SNR] p.138 (21) & (22)
					for (int i = 1; i < N2; i++) ac[idxCore][i] += vv[idxCore][i];//[SNR] p.138, (23)
				}
			});
			if (!bDirect) for (int i = 1; i < N2; i++) un[j][i] = T.Zero;
			for (int r = 0; r < cCores; r++) for (int i = 1; i < N2; i++) p[j][i] += ac[coreOffset + r][i];//[SNR] p.138, (23)
		}
#endif

		void fillAlphaCoeffs()
		{
			int idx = 0;
			for (int k = 1; k <= n; k++)
			{
				int m = 1 << (k - 1);//2ᵏ⁻¹
				T _2ᵏ = T.CreateTruncating(m << 1);
				T _m = T.CreateTruncating(m);
				for (int l = 1; l <= m; l++)
				{
					alphaCoeffs[idx++] = alphaCoeff(l, _2ᵏ, _m);
				}
			}
			T alphaCoeff(int l, T _2ᵏ, T m)
			{
				T sin = T.Sin(T.Pi * T.CreateTruncating(2 * l - 1) / _2ᵏ) / m;
				if (l % 2 == 0) sin = -sin;
				return sin;//[SNR] p.136, (20)
			}
		}

		public override void cleanup()
		{
			p = null;
			alphaCoeffs = null;
			base.cleanup();
		}
	}
}

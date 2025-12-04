#define MeetingProgonka
#define TransposeColumn//32K 14.5sec vs 19.5 sec

using DD128Numeric;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class VariablesSeparationSchemeProgonka<T> : VariablesSeparationScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IHyperbolicFunctions<T>, IPowerFunctions<T>
	{
		T[][] alpha;
		VarSepMethodsEnum method;
		int alfaUB;
#if MeetingProgonka
		int midX;
#endif
#if TransposeColumn
		T[][] uj;
#endif
		T _2 = T.CreateTruncating(2.0);
		T _4 = T.CreateTruncating(4.0);
		T _N2;

		public VariablesSeparationSchemeProgonka(int cXSegments, int cYSegments, T stepX, T stepY, int cCores, Func<T, T, T> fKsi, VarSepMethodsEnum method, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, stepX, stepY, cCores, fKsi, lstBitmap0, fCreateBitmap, reportProgressIn)
		{
#if MeetingProgonka
			if ((N1 & 1) == 0 && (method == VarSepMethodsEnum.Progonka)) throw new System.Exception("VariablesSeparationSchemeProgonka N1 should be odd");
			midX = (N1 - 1) / 2;//N1 should be odd for simplier formulas of meeting progonka
			alfaUB = midX - 1;
#else
			alfaUB = N1 - 2;
#endif
			alpha = new T[cCores][];
			for (int i = 0; i < cCores; i++) alpha[i] = new T[alfaUB + 1];

			progressSteps = 100;//3 loops by 33

			rem1 = N1 / 20;//twice 20(in calculate) plus 60(rem2)
			remPict = N1 / 10;
#if TransposeColumn
			uj = new T[cCores][];
			for (int i = 0; i < cCores; i++) uj[i] = new T[N1 + 1 - 2];
#endif
			this.method = method;
			_N2 = T.CreateTruncating(N2);
		}

		override public T doIteration(int iter)
		{
			initElapsedList();

			float elapsed = getExecutedSeconds(stopWatchEL, () => initRigthHandSide(fKsi, stepX, stepY));
			listElapsedAdd("initRHS", elapsed);

			elapsed = getExecutedSeconds(stopWatchEL, () => transferBoundaryValuesToNearBoundaryNodes());

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2Calculate(T.One, null));//[SNR] p.195, (37), same as [SNR] p.192, (24); using fn[i, k] & fik2i[i, k]
			listElapsedAdd("FFT2_1", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			Action<int> action = null;
			if (method == VarSepMethodsEnum.Progonka) action = progonkaCalculate;
			else if (method == VarSepMethodsEnum.Reduction) action = reduction;
			else if (method == VarSepMethodsEnum.Marching) action = marching;
			elapsed = getExecutedSeconds(stopWatchEL, () => action?.Invoke(60));
			listElapsedAdd(string.Format("{0}", (method == VarSepMethodsEnum.Progonka) ? "Progonka" : (method == VarSepMethodsEnum.Reduction) ? "Reduction" : "Marching"), elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2Calculate(_2 / _N2, addPictureAction));//[SNR] p.195, (39), same as [SNR] p.192, (27); using vk2i[i, k] & fn[i, k]
			listElapsedAdd("FFT2_2", elapsed);

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		public override void cleanup()
		{
			alpha = null;
#if TransposeColumn
			uj = null;
#endif
			base.cleanup();
		}

		void progonkaCalculate(int progressPercent)
		{
			int rem2 = N2 / progressPercent;
			T piDivN2 = T.Pi / _N2;
			T mult = stepX2 / stepY2;//[SNR] p.194, (34);[SNR] p.195, (38)
			AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());

			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>//coreUnordered(made for more cash frendly in low mem cases)
			{
				for (int k2 = 1 + core; k2 < N2; k2 += cCores)
				{
					int k = calcAlpha((T.One + mult * (T.One - T.Cos(piDivN2 * T.CreateTruncating(k2)))) * _2, alpha[core]);//[SNR] p.194, (34);sine of half angle
					progonka(core, k2, k);//[SNR] p.195, (38);fik2i == vk2i == un

					if ((rem2 > 0) && (k2 % rem2 == 0)) showProgress();
					if (loopState.IsStopped) return;
					if (areIterationsCanceled())
					{
						loopState.Stop();
						return;
					}
				}
			});

			int calcAlpha(T diagElem, T[] alfa)
			{
				int loopUpper = αCC.upperBound(diagElem, alfaUB);
				int k = loopUpper;
				alfa[0] = T.One / diagElem;//[SNR] p.195(40)
				for (int i = 1; i <= loopUpper; i++)
				{
					alfa[i] = T.One / (diagElem - alfa[i - 1]);//[SNR] p.195(40)
				}
				return k;
			}

			void progonka(int idxAlpha, int colRes, int k)
			{//beta array is not used - it is placed to un
				T[] alfa = alpha[idxAlpha];
				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				int ind(int idx) => (idx < k) ? idx : k;
#if MeetingProgonka
#if !TransposeColumn
				un[1, colRes] = (stepX2 * un[1, colRes] + 0) * alfa[ind(0)];
				for (int i = 1; i < midX; i++) un[i + 1, colRes] = (stepX2 * un[i + 1, colRes] + un[i - 0, colRes]) * alfa[ind(i)];//[SNR] p.195(40);SNR p.77(12,formula 2)

				int U = N1 - 2;
				un[U + 1, colRes] = (stepX2 * un[U + 1, colRes] + 0) * alfa[ind(U - U)];
				for (int i = U - 1; i > midX - 1; i--) un[i + 1, colRes] = (stepX2 * un[i + 1, colRes] + un[i + 2, colRes]) * alfa[ind(U - i)];//[SNR] p.195(40);SNR p.77(12,formula 4)

				un[midX + 1, colRes] = (un[midX + 1, colRes] + alfa[ind(midX - 1)] * un[midX, colRes]) / (1 - alfa[ind(midX - 1)] * alfa[ind(midX - 1)]);

				for (int i = midX - 1; i >= 0; i--) un[i + 1, colRes] += alfa[ind(i)] * un[i + 2, colRes];
				for (int i = midX + 1; i <= U; i++) un[i + 1, colRes] += alfa[ind(U - i)] * un[i - 0, colRes];
#else
				T[] ut = uj[idxAlpha];
				for (int i = 0; i <= N1 - 2; i++) ut[i] = un[i + 1][colRes];

				int U = N1 - 2;
				for (int i = 0; i <= midX - 1; i++)
				{
					ut[i + 0] = (stepX2 * ut[i + 0] + (i != 0 ? ut[i - 1 + 0] : T.Zero)) * alfa[ind(i)];//[SNR] p.195(40);SNR p.77(12,formula 2)
					ut[U - i] = (stepX2 * ut[U - i] + (i != 0 ? ut[U - i + 1] : T.Zero)) * alfa[ind(i)];//[SNR] p.195(40);SNR p.77(12,formula 4)
				}

				ut[midX] = (ut[midX] + alfa[ind(midX - 1)] * ut[midX - 1]) / (T.One - alfa[ind(midX - 1)] * alfa[ind(midX - 1)]);

				for (int i = midX - 1; i >= 0; i--) ut[i] += alfa[ind(i)] * ut[i + 1];
				for (int i = midX + 1; i <= U; i++) ut[i] += alfa[ind(U - i)] * ut[i - 1];

				for (int i = 0; i <= N1 - 2; i++) un[i + 1][colRes] = ut[i];
#endif
#else
			//using 0 instead of un[0, colRes] which is NOT zero(boundary value), but assumed to be zero(transferred value to near boundary node)
			un[1, colRes] = (stepX2 * un[1, colRes]) * alfa[0];
			for (int i = 2; i <= N1 - 1; i++) un[i, colRes] = (stepX2 * un[i, colRes] + un[i - 1, colRes]) * alfa[i - 1];//[SNR] p.195(40);

			//using 0 instead of un[N1, colRes] which is NOT zero(boundary value), but assumed to be zero(transferred value to near boundary node)
			//not adding zero value for un[N1 - 1, colRes] += alfa[N1 - 2] * un[N1 - 0, colRes]; because un[N1 - 0, colRes] == 0
			for (int i = N1 - 2; i >= 1; i--) un[i, colRes] += alfa[i - 1] * un[i + 1, colRes];//[SNR] p.195(40);
#endif
			}
		}

		void reduction(int progressPercent)//CPU_double
		{
			int rem2 = (N2 - 1) / progressPercent;
			T pi2N2 = T.Pi / _N2 / _2;
			T mult = stepX2 * _4 / stepY2;//[SNR] p.194, (34);[SNR] p.195, (38)
			int cCores = optionsParallel.MaxDegreeOfParallelism;
			T[,] a = new T[cCores, N1 + 1];
			T[,] b = new T[cCores, N1 + 1];
			T[,] d = new T[cCores, N1 + 1];
			T[,] f = new T[cCores, N1 + 1];

			int m = (int)Math.Ceiling(Math.Log(N1, 2));//[Nik] p32
			int[] n = new int[m];

			calculateNArray();

			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
			{
				for (int k2 = 1 + core; k2 < N2; k2 += cCores)
				{
					T sin = T.Sin(pi2N2 * T.CreateTruncating(k2));
					forwardPath(mult * sin * sin + _2, core, k2);
					middlePoint(core, k2);
					reversePath(core, k2);

					if ((rem2 > 0) && (k2 % rem2 == 0)) showProgress();
					if (loopState.IsStopped) return;
					if (areIterationsCanceled())
					{
						loopState.Stop();
						return;
					}
				}
			});

			void calculateNArray()
			{
				n[0] = N1 - 1;
				for (int k = 0; k <= m - 2; k++) n[k + 1] = n[k] / 2;
			}

			void forwardPath(T diagElem, int core, int colRes)
			{
				T α;
				for (int i = 1; i <= N1; i++) a[core, i] = T.One;//a[0] == 0
				for (int i = 0; i < N1; i++) b[core, i] = T.One;//b[N] == 0
				for (int i = 0; i <= N1; i++) d[core, i] = diagElem - a[core, i] - b[core, i];
				for (int i = 1; i < N1; i++) f[core, i] = stepX2 * un[i][colRes];//rightHandSide from [SNR] p.195, (38)
				//f[core, 0] = un[0][colRes];
				//f[core, N1] = un[N1][colRes];

				for (int k = 0; k <= m - 2; k++)//forward path
				{
					int i1 = 1 << k;
					for (int j = 1; j <= n[k + 1]; j++)
					{
						int i = (1 << (k + 1)) * j;
						int im = i - i1;//i minus
						int ip = i + i1;//i plus
						α = a[core, i] / (a[core, im] + d[core, im] + b[core, im]);
						a[core, i] = α * a[core, im];
						if (j == n[k + 1] && n[k] % 2 == 0)
						{
							d[core, i] += α * d[core, im];
							f[core, i] += α * f[core, im];
						}
						else
						{
							T β = b[core, i] / (a[core, ip] + d[core, ip] + b[core, ip]);
							b[core, i] = β * b[core, ip];

							d[core, i] += α * d[core, im] + β * d[core, ip];
							f[core, i] += α * f[core, im] + β * f[core, ip];
						}
					}
				}
			}

			void middlePoint(int core, int colRes)
			{
				int i = 1 << (m - 1);
				un[i][colRes] = f[core, i] / (a[core, i] + d[core, i] + b[core, i]);
			}

			void reversePath(int core, int colRes)
			{
				for (int k = m - 2; k >= 0; k--)//reverse path
				{
					int i1 = 1 << k;
					int i = i1 * 1;

					un[i][colRes] = (f[core, i] + b[core, i] * un[i + i1][colRes]) / (a[core, i] + d[core, i] + b[core, i]);

					for (int j = 3; j <= 2 * n[k + 1] - 1; j += 2)
					{
						i = i1 * j;
						un[i][colRes] = (f[core, i] + a[core, i] * un[i - i1][colRes] + b[core, i] * un[i + i1][colRes]) / (a[core, i] + d[core, i] + b[core, i]);
					}

					if (n[k] % 2 == 1)
					{
						i = i1 * n[k];
						un[i][colRes] = (f[core, i] + a[core, i] * un[i - i1][colRes]) / (a[core, i] + d[core, i] + b[core, i]);
					}
				}
			}
		}

		void marching(int progressPercent)
		{
			int rem2 = N2 / progressPercent;
			T piDivN2 = T.Pi / _N2;
			T mult = stepX2 / stepY2;//[SNR] p.194, (34);[SNR] p.195, (38)
			T[][] rhs = new T[cCores][];
			for (int i = 0; i < cCores; i++) rhs[i] = new T[N1];//zero index not used

			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>//coreUnordered(made for more cash frendly in low mem cases)
			{
				for (int k2 = 1 + core; k2 < N2; k2 += cCores)
				{
					T diagElem = (T.One + mult * (T.One - T.Cos(piDivN2 * T.CreateTruncating(k2)))) * _2;
					calculate(diagElem, k2, core);

					if ((rem2 > 0) && (k2 % rem2 == 0)) showProgress();
					if (loopState.IsStopped) return;
					if (areIterationsCanceled())
					{
						loopState.Stop();
						return;
					}
				}
			});

			void calculate(T diagElem, int colRes, int core)
			{
				int k, L;//k cannot be less 2
				T[] p0, p1, p2;
				T[] q0, q1, q2;
				T a0, a1, a2, a, b;
				T[] fi;

				for (int i = 1; i <= N1 - 1; i++) rhs[core][i] = stepX2 * un[i][colRes];

				T y0 = T.Zero;//not un[0][colRes] because of transferBoundaryValuesToNearBoundaryNodes
				T yN = T.Zero;//not un[N1][colRes] because of transferBoundaryValuesToNearBoundaryNodes

				int[] paramsL = UtilsLParam.getMarchingLParamArray(N1);
				L = paramsL[paramsL.Length - 1];
				k = (N1 - 1) / (2 * L);

				p0 = new T[L];
				p1 = new T[L];
				p2 = new T[L];

				q0 = new T[L];
				q1 = new T[L];
				q2 = new T[L];

				fi = new T[2 * L];

				calculatePl();
				calculateQl();
				calculateA();

				calculateFi();
				progonkaNonmonotonic();

				calculateResult();

				//local functions
				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				T rightHandSide(int i) => rhs[core][i];

				void calculatePl()
				{
					for (int l = 0; l < L; l++)
					{
						int offs = 2 * l * k;
						p0[l] = T.Zero;//T.NaN;
						p1[l] = T.Zero;
						p2[l] = rightHandSide(offs + 1);
						for (int j = 2; j <= k; j++)//SNE p.233(32)
						{
							p0[l] = p1[l];
							p1[l] = p2[l];
							p2[l] = diagElem * p1[l] - p0[l] + rightHandSide(offs + j);
						}
					}
				}

				void calculateQl()
				{
					for (int l = 0; l < L; l++)
					{
						int offs = 2 * (l + 1) * k;
						q0[l] = T.Zero;//T.NaN;
						q1[l] = T.Zero;
						q2[l] = rightHandSide(offs);
						for (int j = 2; j <= k; j++)//SNE p.233(32)
						{
							q0[l] = q1[l];
							q1[l] = q2[l];
							q2[l] = diagElem * q1[l] - q0[l] + rightHandSide(offs + 1 - j);
						}
					}
				}

				void calculateA()
				{
					a0 = T.Zero;//T.NaN;
					a1 = T.One;
					a2 = diagElem;
					for (int j = 2; j <= k; j++)//SNE p.229(17)
					{
						a0 = a1;
						a1 = a2;
						a2 = diagElem * a1 - a0;
					}
					b = T.One / (a1 * a1 - a0 * a0);//SNE p.233(34)
					a = a1 * (a2 - a0) * b;//SNE p.233(34)
#if DEBUG2
					T stability = Math.Abs(a) - 1.0 - Math.Abs(b);
					if (stability < 0)
					{
						Trace.WriteLine(string.Format("|a| - 1 - |b|={0}, stable {1}", stability, stability > 0 ? "yes" : "no"));
					}
#endif
				}

				void calculateFi()
				{
					for (int l = 0; l < L; l++)//SNE p.233(34)
					{
						T g1 = q1[l] - p2[l];//SNE p.232(31)
						T g2 = p1[l] - q2[l];//SNE p.232(31)
						fi[2 * l + 1 - 1] = -(a1 * g1 + a0 * g2) * b;//easier formulas than SNE p.233(34)
						fi[2 * l + 2 - 1] = -(a0 * g1 + a1 * g2) * b;//get by matrix(above (33)) multiplication by right hand side of (33)
					}
					fi[0] += y0;
					fi[fi.GetUpperBound(0)] += yN;
				}

				void progonkaNonmonotonic()
				{//SNR p.96 middle
				 //make arrays sizes less 1 because α[0],β[0],θ[0],χ[0] is not used in the original algorithm(except χ[0] which is 0)
				 //for L = 1, system for w is
				 //used case
				 //+a -b = fi1-(-1*y0)
				 //-b +a = fi2-(-1*yN)
				 //was get from system
				 // 1  0       = fi0==y0
				 //-1  a -b    = fi1
				 //   -b  a -1 = fi2
				 //       0  1 = fi3==yN
					int M = fi.GetUpperBound(0);
					T[] α = new T[M];
					T[] β = new T[M];
					int[] θ = new int[M];
					int[] χ = new int[M];

					T C = a, A = aa(1), F = fi[0], Frus = fi[1];//SNR p.95, after (53)

					T bb(int i) => (i % 2 == 0) ? b : T.One;//over diagonal
					T aa(int i) => (i % 2 == 1) ? b : T.One;//under diagonal

					for (int i = 0; i <= M - 1; i++)
					{
						if (T.Abs(C) >= T.Abs(bb(i)))
						{
							α[i] = bb(i) / C;
							β[i] = F / C;
							C = a - A * α[i];
							F = Frus + A * β[i];
							θ[i] = (i != 0) ? χ[i - 1] : 0;
							χ[i] = i + 1;
							if (i != M - 1)
							{
								A = aa(i + 2);
								Frus = fi[i + 2];
							}
						}
						else
						{
							α[i] = C / bb(i);
							β[i] = -F / bb(i);
							C = a * α[i] - A;
							F = Frus - a * β[i];
							θ[i] = i + 1;
							χ[i] = (i != 0) ? χ[i - 1] : 0;
							if (i != M - 1)
							{
								A = aa(i + 2) * α[i];
								Frus = fi[i + 2] + aa(i + 2) * β[i];
							}
						}
					}

					un[wToUn(χ[M - 1] + 0)][colRes] = F / C;
					for (int i = M - 1; i >= 0; i--)
					{
						un[wToUn(θ[i] + 0)][colRes] = α[i] * un[wToUn(χ[i] + 0)][colRes] + β[i];
					}
				}

				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				int wToUn(int idx)
				{
					//Trace.WriteLine(string.Format("{0} -> {1}", idx, idx * k + (idx % 2 == 0 ? 1 : k)));
					return idx * k + (idx % 2 == 0 ? 1 : k);
				}

				//-->      <-- -->     <-- -->      <--
				//0,1        2,3         4,5        6,7
				//|_____0_____|_____1_____|_____2_____|
				void calculateResult()
				{
					T z0 = un[0][colRes];
					T zN = un[N1][colRes];
					un[0][colRes] = y0;
					un[N1][colRes] = yN;
					for (int l = 0; l < L; l++)
					{
						for (int j = 2 * (l * k + 1); j <= (2 * l + 1) * k; j++) un[j][colRes] = diagElem * un[j - 1][colRes] - un[j - 2][colRes] - rightHandSide(j - 1);//SNE p.232(30)
						for (int j = 2 * (l + 1) * k - 1; j >= (2 * l + 1) * k + 1; j--) un[j][colRes] = diagElem * un[j + 1][colRes] - un[j + 2][colRes] - rightHandSide(j + 1);//SNE p.232(30)
					}
					un[0][colRes] = z0;
					un[N1][colRes] = zN;
				}
			}
		}
	}
}

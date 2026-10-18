//#define MeetingProgonka
#define TransposeColumn//32K 14.5sec vs 19.5 sec

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
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
	class VariablesSeparationSchemeProgonka<T> : VariablesSeparationScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IHyperbolicFunctions<T>, IPowerFunctions<T>, IExponentialFunctions<T>
	{
		T _2 = T.CreateTruncating(2.0);
		T _N2;
		VSSolver solver = null;

		public VariablesSeparationSchemeProgonka(RectangleData<T> rectData, Func<T, T, T> fKsi, VarSepMethodsEnum method, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(rectData, fKsi, lstBitmap0, fCreateBitmap, reportProgressIn)
		{
			int cCores = GridIterator.optionsParallel.MaxDegreeOfParallelism;

			progressSteps = 100;//3 loops by 33

			rem1 = N1 / 20;//twice 20(in calculate) plus 60(rem2)
			remPict = N1 / 10;
			_N2 = T.CreateTruncating(N2);
			if (method == VarSepMethodsEnum.Progonka) solver = new Progonka(un, N1, _N2, stepX2, stepY2, cCores);
			else if (method == VarSepMethodsEnum.Reduction) solver = new Reduction(un, N1, _N2, stepX2, stepY2, cCores);
			else if (method == VarSepMethodsEnum.Marching) solver = new Marching(un, N1, _N2, stepX2, stepY2, cCores);
		}

		override public T doIteration(int iter)
		{
			initElapsedList();

			float elapsed = getExecutedSeconds(stopWatchEL, () => initRigthHandSide(fKsi, stepX, stepY));
			listElapsedAdd("initRHS", elapsed);

			//UtilsPrint.printJaggedArray(un, "{0,10:0.000}", "before transfer");
			elapsed = getExecutedSeconds(stopWatchEL, () => transferBoundaryValuesToNearBoundaryNodes());

			//UtilsPrint.printJaggedArray(un, "{0,10:0.000}", "before fftN2Calculate");
			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2Calculate(T.One, null));//[SNR] p.195, (37), same as [SNR] p.192, (24); using fn[i, k] & fik2i[i, k]
			//UtilsPrint.printJaggedArray(un, "{0,10:0.000}", "after fftN2Calculate");
			listElapsedAdd("FFT2_1", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => runSolvers(60, solver));
			//UtilsPrint.printJaggedArray(un, "{0,10:0.000}", "after runSolvers");
			listElapsedAdd(string.Format("{0}", solver.name()), elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2Calculate(_2 / _N2, addPictureAction));//[SNR] p.195, (39), same as [SNR] p.192, (27); using vk2i[i, k] & fn[i, k]
			listElapsedAdd("FFT2_2", elapsed);
			//UtilsPrint.printJaggedArray(un, "{0,7:0.000}", nameof(action.Method.Name));

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		public override void cleanup()
		{
			base.cleanup();
		}

		void runSolvers(int progressPercent, VSSolver solver)
		{
			int rem2 = N2 / progressPercent;

			int cCores = GridIterator.optionsParallel.MaxDegreeOfParallelism;
			Parallel.For(0, cCores, GridIterator.optionsParallel, (core, loopState) =>//coreUnordered(made for more cash frendly in low mem cases)
			{
				for (int k2 = 1 + core; k2 < N2; k2 += cCores)
				{
					solver.calculate(core, k2);

					if ((rem2 > 0) && (k2 % rem2 == 0)) showProgress();
					if (loopState.IsStopped) return;
					if (areIterationsCanceled())
					{
						loopState.Stop();
						return;
					}
				}
			});
		}

		class Progonka : VSSolver
		{
			T[][] alpha, un;
			int N1, alfaUB;
			T _N2, stepX2, stepY2;
			T piDivN2;
			T mult;//[SNR] p.194, (34);[SNR] p.195, (38)
			AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
#if MeetingProgonka
			int midX;
#endif
#if TransposeColumn
			T[][] uj;
#endif
			static T _2 = T.CreateTruncating(2.0);

			public Progonka(T[][] un, int N1, T _N2, T stepX2, T stepY2, int cCores)
			{
#if MeetingProgonka
				if ((N1 & 1) == 0) throw new System.Exception("VariablesSeparationSchemeProgonka N1 should be odd");
				midX = (N1 - 1) / 2;//N1 should be odd for simplier formulas of meeting progonka
				alfaUB = midX - 1;
#else
				alfaUB = N1 - 2;
#endif
				this.N1 = N1;
				this._N2 = _N2;
				this.stepX2 = stepX2;
				this.stepY2 = stepY2;
				this.un = un;

				alpha = new T[cCores][];
				for (int i = 0; i < cCores; i++) alpha[i] = new T[alfaUB + 1];

				piDivN2 = T.Pi / _N2;
				mult = stepX2 / stepY2;//[SNR] p.194, (34);[SNR] p.195, (38)
#if TransposeColumn
				uj = new T[cCores][];
				for (int i = 0; i < cCores; i++) uj[i] = new T[N1 + 1 - 2];
#endif
			}

			public void calculate(int core, int k2)
			{
				//Debug.Write(string.Format("mult {0} piDivN2 {1} N2 {2} k2 {3}", mult, piDivN2, _N2, k2));
				int k = calcAlpha((T.One + mult * (T.One - T.Cos(piDivN2 * T.CreateTruncating(k2)))) * _2, alpha[core]);//[SNR] p.194, (34);sine of half angle
				progonka(core, k2, k);//[SNR] p.195, (38);fik2i == vk2i == un
			}
			public string name() => "Progonka";

			int calcAlpha(T diagElem, T[] alfa)
			{
				int k = αCC.upperBound(diagElem, alfaUB);
				Debug.WriteLine(string.Format(CultureInfo.InvariantCulture, "diagElem {0,16:0.00000000000000} k {1,2}", diagElem, k));
				alfa[0] = T.One / diagElem;//[SNR] p.195(40)
				for (int i = 1; i <= k; i++)
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
			//using 0 instead of un[0][colRes] which is NOT zero(boundary value), but assumed to be zero(transferred value to near boundary node)
			un[1][colRes] = (stepX2 * un[1][colRes]) * alfa[ind(0)];
			for (int i = 2; i <= N1 - 1; i++) un[i][colRes] = (stepX2 * un[i][colRes] + un[i - 1][colRes]) * alfa[ind(i - 1)];//[SNR] p.195(40);

			//using 0 instead of un[N1][colRes] which is NOT zero(boundary value), but assumed to be zero(transferred value to near boundary node)
			//not adding zero value for un[N1 - 1][colRes] += alfa[ind(N1 - 2)] * un[N1 - 0][colRes]; because un[N1 - 0][colRes] == 0
			for (int i = N1 - 2; i >= 1; i--) un[i][colRes] += alfa[ind(i - 1)] * un[i + 1][colRes];//[SNR] p.195(40);
#endif
			}
		}

		class Reduction : VSSolver
		{
			readonly T[][] un;
			readonly int N1, m;
			readonly T _N2, stepX2, stepY2;
			readonly int[] n;
			readonly T[][] a, b, d, f;
			readonly static T _2 = T.CreateTruncating(2.0), _4 = T.CreateTruncating(4.0);
			readonly T pi2N2, mult;

			public Reduction(T[][] un, int N1, T _N2, T stepX2, T stepY2, int cCores)
			{
				this.N1 = N1;
				this._N2 = _N2;
				this.stepX2 = stepX2;
				this.stepY2 = stepY2;
				this.un = un;

				pi2N2 = T.Pi / _N2 / _2;
				mult = stepX2 * _4 / stepY2;//[SNR] p.194, (34);[SNR] p.195, (38)

				a = new T[cCores][];
				b = new T[cCores][];
				d = new T[cCores][];
				f = new T[cCores][];
				for (int i = 0; i < cCores; i++)
				{
					a[i] = new T[N1 + 1];
					b[i] = new T[N1 + 1];
					d[i] = new T[N1 + 1];
					f[i] = new T[N1 + 1];
				}

				m = (int)Math.Ceiling(Math.Log(N1, 2));//[Nik] p32
				n = new int[m];
				calculateNArray();
			}

			public void calculate(int core, int k2)
			{
				T sin = T.Sin(pi2N2 * T.CreateTruncating(k2));
				forwardPath(mult * sin * sin + _2, a[core], b[core], d[core], f[core], k2);
				middlePoint(a[core], b[core], d[core], f[core], k2);
				reversePath(a[core], b[core], d[core], f[core], k2);
			}
			public string name() => "Reduction";

			void calculateNArray()
			{
				n[0] = N1 - 1;
				for (int k = 0; k <= m - 2; k++) n[k + 1] = n[k] / 2;
			}

			void forwardPath(T diagElem, T[] a, T[] b, T[] d, T[] f, int colRes)
			{
				T α;
				void actionInit(int i)
				{
					a[i] = b[i] = T.One;
					d[i] = diagElem - a[i] - b[i];
					if (i != 0 && i != N1) f[i] = stepX2 * un[i][colRes];
				}
				GridIterator.iterate(0, N1, actionInit);
				a[0] = T.Zero;
				d[0] = diagElem - a[0] - b[0];
				b[N1] = T.Zero;
				d[N1] = diagElem - a[N1] - b[N1];

				for (int k = 0; k <= m - 2; k++)//forward path
				{
					int i1 = 1 << k;
					for (int j = 1; j <= n[k + 1]; j++)
					{
						int i = (1 << (k + 1)) * j;
						int im = i - i1;//i minus
						int ip = i + i1;//i plus
						α = a[i] / (a[im] + d[im] + b[im]);
						a[i] = α * a[im];
						if (j == n[k + 1] && n[k] % 2 == 0)
						{
							d[i] += α * d[im];
							f[i] += α * f[im];
						}
						else
						{
							T β = b[i] / (a[ip] + d[ip] + b[ip]);
							b[i] = β * b[ip];

							d[i] += α * d[im] + β * d[ip];
							f[i] += α * f[im] + β * f[ip];
						}
					}
				}
			}

			void middlePoint(T[] a, T[] b, T[] d, T[] f, int colRes)
			{
				int i = 1 << (m - 1);
				un[i][colRes] = f[i] / (a[i] + d[i] + b[i]);
			}

			void reversePath(T[] a, T[] b, T[] d, T[] f, int colRes)
			{
				for (int k = m - 2; k >= 0; k--)//reverse path
				{
					int i1 = 1 << k;
					int i = i1 * 1;

					un[i][colRes] = (f[i] + b[i] * un[i + i1][colRes]) / (a[i] + d[i] + b[i]);

					for (int j = 3; j <= 2 * n[k + 1] - 1; j += 2)
					{
						i = i1 * j;
						un[i][colRes] = (f[i] + a[i] * un[i - i1][colRes] + b[i] * un[i + i1][colRes]) / (a[i] + d[i] + b[i]);
					}

					if (n[k] % 2 == 1)
					{
						i = i1 * n[k];
						un[i][colRes] = (f[i] + a[i] * un[i - i1][colRes]) / (a[i] + d[i] + b[i]);
					}
				}
			}
		}

		class Marching : VSSolver
		{
			readonly T[][] un;
			readonly int N1, k, L, M;
			readonly T _N2;
			readonly T piDivN2, mult, stepX2, stepY2;
			readonly T[][] p0, p1, p2;
			readonly T[][] q0, q1, q2;
			readonly T[][] rhs;
			readonly T[] a0, a1, a2, a, b;
			readonly T[][] fi;
			readonly T[][] α, β;
			readonly int[][] θ, χ;
			readonly T y0 = T.Zero, yN = T.Zero;//not un[0][colRes], un[N1][colRes] because of transferBoundaryValuesToNearBoundaryNodes
			readonly static T _2 = T.CreateTruncating(2.0);

			public Marching(T[][] un, int N1, T _N2, T stepX2, T stepY2, int cCores)
			{
				this.un = un;
				this.N1 = N1;
				this._N2 = _N2;
				this.stepX2 = stepX2;
				this.stepY2 = stepY2;

				piDivN2 = T.Pi / _N2;
				mult = stepX2 / stepY2;//[SNR] p.194, (34);[SNR] p.195, (38)

				int[] paramsL = UtilsLParam.getMarchingLParamArray(N1);
				L = paramsL[paramsL.Length - 1];
				k = (N1 - 1) / (2 * L);

				p0 = new T[cCores][];
				p1 = new T[cCores][];
				p2 = new T[cCores][];

				q0 = new T[cCores][];
				q1 = new T[cCores][];
				q2 = new T[cCores][];

				fi = new T[cCores][];
				rhs = new T[cCores][];

				α = new T[cCores][];
				β = new T[cCores][];
				θ = new int[cCores][];
				χ = new int[cCores][];

				a0 = new T[cCores];
				a1 = new T[cCores];
				a2 = new T[cCores];
				a = new T[cCores];
				b = new T[cCores];

				M = 2 * L - 1;//fi[0].GetUpperBound(0);
				for (int i = 0; i < cCores; i++)
				{
					p0[i] = new T[L];
					p1[i] = new T[L];
					p2[i] = new T[L];
					q0[i] = new T[L];
					q1[i] = new T[L];
					q2[i] = new T[L];

					fi[i] = new T[M + 1];
					rhs[i] = new T[N1];//zero index not used

					α[i] = new T[M];
					β[i] = new T[M];
					θ[i] = new int[M];
					χ[i] = new int[M];
				}
			}

			public void calculate(int core, int k2)
			{
				T diagElem = (T.One + mult * (T.One - T.Cos(piDivN2 * T.CreateTruncating(k2)))) * _2;
				for (int i = 1; i <= N1 - 1; i++) rhs[core][i] = stepX2 * un[i][k2];

				calculatePl(diagElem, p0[core], p1[core], p2[core], rhs[core]);
				calculateQl(diagElem, q0[core], q1[core], q2[core], rhs[core]);
				calculateA(diagElem, core);

				calculateFi(p1[core], p2[core], q1[core], q2[core], fi[core], a0[core], a1[core], b[core]);
				progonkaNonmonotonic(α[core], β[core], θ[core], χ[core], fi[core], a[core], b[core], k2);

				calculateResult(diagElem, rhs[core], k2);
			}
			public string name() => "Marching";

			void calculatePl(T diagElem, T[] p0, T[] p1, T[] p2, T[] rhs)
			{
				for (int l = 0; l < L; l++)
				{
					int offs = 2 * l * k;
					p0[l] = T.Zero;//T.NaN;
					p1[l] = T.Zero;
					p2[l] = rhs[offs + 1];
					for (int j = 2; j <= k; j++)//SNE p.233(32)
					{
						p0[l] = p1[l];
						p1[l] = p2[l];
						p2[l] = diagElem * p1[l] - p0[l] + rhs[offs + j];
					}
				}
			}

			void calculateQl(T diagElem, T[] q0, T[] q1, T[] q2, T[] rhs)
			{
				for (int l = 0; l < L; l++)
				{
					int offs = 2 * (l + 1) * k;
					q0[l] = T.Zero;//T.NaN;
					q1[l] = T.Zero;
					q2[l] = rhs[offs];
					for (int j = 2; j <= k; j++)//SNE p.233(32)
					{
						q0[l] = q1[l];
						q1[l] = q2[l];
						q2[l] = diagElem * q1[l] - q0[l] + rhs[offs + 1 - j];
					}
				}
			}

			void calculateA(T diagElem, int core)
			{
				a0[core] = T.Zero;//T.NaN;
				a1[core] = T.One;
				a2[core] = diagElem;
				for (int j = 2; j <= k; j++)//SNE p.229(17)
				{
					a0[core] = a1[core];
					a1[core] = a2[core];
					a2[core] = diagElem * a1[core] - a0[core];
				}
				b[core] = T.One / (a1[core] * a1[core] - a0[core] * a0[core]);//SNE p.233(34)
				a[core] = a1[core] * (a2[core] - a0[core]) * b[core];//SNE p.233(34)
#if DEBUG2
					T stability = Math.Abs(a) - 1.0 - Math.Abs(b);
					if (stability < 0)
					{
						Trace.WriteLine(string.Format("|a| - 1 - |b|={0}, stable {1}", stability, stability > 0 ? "yes" : "no"));
					}
#endif
			}

			void calculateFi(T[] p1, T[] p2, T[] q1, T[] q2, T[] fi, T a0, T a1, T b)
			{
				for (int l = 0; l < L; l++)//SNE p.233(34)
				{
					T g1 = q1[l] - p2[l];//SNE p.232(31)
					T g2 = p1[l] - q2[l];//SNE p.232(31)
					fi[2 * l + 1 - 1] = -(a1 * g1 + a0 * g2) * b;//easier formulas than SNE p.233(34)
					fi[2 * l + 2 - 1] = -(a0 * g1 + a1 * g2) * b;//get by matrix(above (33)) multiplication by right hand side of (33)
				}
				fi[0] += y0;
				fi[M] += yN;
			}

			void progonkaNonmonotonic(T[] α, T[] β, int[] θ, int[] χ, T[] fi, T a, T b, int colRes)
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
			void calculateResult(T diagElem, T[] rhs, int colRes)
			{
				T z0 = un[0][colRes];
				T zN = un[N1][colRes];
				un[0][colRes] = y0;
				un[N1][colRes] = yN;
				for (int l = 0; l < L; l++)
				{
					for (int j = 2 * (l * k + 1); j <= (2 * l + 1) * k; j++) un[j][colRes] = diagElem * un[j - 1][colRes] - un[j - 2][colRes] - rhs[j - 1];//SNE p.232(30)
					for (int j = 2 * (l + 1) * k - 1; j >= (2 * l + 1) * k + 1; j--) un[j][colRes] = diagElem * un[j + 1][colRes] - un[j + 2][colRes] - rhs[j + 1];//SNE p.232(30)
				}
				un[0][colRes] = z0;
				un[N1][colRes] = zN;
			}
		}

		interface VSSolver
		{
			void calculate(int core, int k2);
			string name();
		}
	}
}

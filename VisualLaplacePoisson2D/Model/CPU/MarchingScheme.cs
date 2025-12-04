#define MeetingProgonka

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class MarchingScheme<T> : DirectJaggedScheme<T>, IScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>
	{
		protected T[][] fn;
		float[][] unShow;
		Func<T, T, T> fKsi;
		T[][] pq0, pq1, pq2, ŵ;//p & q arrays in one array
		readonly int k, L, M, cCores, pqDim2;
		T stepX, stepY, cBase, ai, bi, subsupra, normFactor;//subsupra - subdiagonal,supradiagonal
		protected List<BitmapSource> lstBitmap;
		Action<double> reportProgress;
		float curProgress, oldProgress;
		bool iterationsCanceled;
		CancellationTokenSource cts = new CancellationTokenSource();
		readonly ParallelOptions optionsParallel;
		FFTCalculator<T> fftM;
		T km1, k0;
		T _2 = T.CreateTruncating(2);
		T _4 = T.CreateTruncating(4);

		public MarchingScheme(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsiIn, int paramL, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, fCreateBitmap)
		{
			fn = new T[Nx - 1][];
			for (int i = 0; i < Nx - 1; i++) fn[i] = new T[Ny + 1];

			L = paramL;//max value == (Nx - 1) / 4, because k is minimum 2
			k = (Nx - 1) / (2 * L);//k > 1 always
			km1 = T.CreateTruncating(k - 1);
			k0 = T.CreateTruncating(k);

			M = Ny - 1;
			pqDim2 = M + 2;//'+ 2' for use with un array; assumes M = Ny - 1

			//Bank Rose Marching Algorithms for Elliptic Boundary Value Problems. I The Constant Coefficient Case.pdf p.21(6.8)
			normFactor = T.Sqrt(T.CreateTruncating(2.0 / Ny));//normalizing factor;Demmel rus 281

			fftM = new FFTCalculator<T>(cCores, Ny);

			stepX = stepXIn;
			stepY = stepYIn;
			subsupra = stepX * stepX / (stepY * stepY);//[SNR] p.106, (4)
			ai = subsupra;//subdiagonal,[SNR] p.106, (4)
			bi = subsupra;//supradiagonal,[SNR] p.106, (4)

			fKsi = fKsiIn;

			lstBitmap = lstBitmap0;
			if (lstBitmap != null)
			{
				unShow = new float[Nx + 1][];
				for (int i = 0; i < Nx + 1; i++) unShow[i] = new float[Ny + 1];
			}

			reportProgress = reportProgressIn;
			curProgress = 0;
			oldProgress = 0;

			cBase = (T.One + subsupra) * _2;//[SNR] p.106, (4)

			this.cCores = cCores;
			optionsParallel = new ParallelOptions() { MaxDegreeOfParallelism = cCores, CancellationToken = cts.Token };
		}

		public T doIteration(int iter)
		{
			initElapsedList();
			float elapsed = getExecutedSeconds(stopWatchEL, () => initFj());
			listElapsedAdd("initFj", elapsed);

			forwardPath();
			if (iterationsCanceled) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => solveReducedSystem());
			listElapsedAdd("reducedSystem", elapsed);
			if (iterationsCanceled) return T.Zero;

			reversePath();
			if (iterationsCanceled) return T.Zero;

			restoreBounds(un);

			return T.Zero;
		}

		void initFj()
		{
			T stepX2 = stepX * stepX;
			if (fKsi != null) GridIterator.iterate(0, Nx - 1, 1, fn[0].GetUpperBound(0), (i, j) => { fn[i][j] += stepX2 * fKsi(stepX * T.CreateTruncating(i + 1), stepY * T.CreateTruncating(j)); });//[SNR] p.123 (8),[SNE] p.120 (8)

			Parallel.For(0, Nx - 1, optionsParallel, (i) =>
			{
				fn[i][1] += subsupra * bndB[i];//[SNR] p.123 (8),[SNE] p.120 (8)
				fn[i][Ny - 1] += subsupra * bndT[i];//[SNR] p.123 (8),[SNE] p.120 (8)
			});

			Parallel.For(1, Ny, optionsParallel, (j) =>
			{
				fn[0][j] += bndL[j - 1];//analog [SNR] p.123 (8),[SNE] p.120 (8)
				fn[Nx - 2][j] += bndR[j - 1];//analog [SNR] p.123 (8),[SNE] p.120 (8)
			});
		}

		void forwardPath()
		{
			float elapsed = getExecutedSeconds(stopWatchEL, () => calculatePQVectors());
			listElapsedAdd("calculatePQVectors", elapsed);
			if (iterationsCanceled) return;
			elapsed = getExecutedSeconds(stopWatchEL, () => calculateĜ());
			listElapsedAdd("calculateĜ", elapsed);
		}

		void solveReducedSystem()
		{
			T[][] _φ = new T[cCores][];
			T[][] _α = new T[cCores][];
			ŵ = pq0;
#if MeetingProgonka
			int alfaUB = L - 1;//alfa upper bounds
#else
			int alfaUB = 2 * L - 1;//alfa upper bounds
#endif
			int U = 2 * L - 1;//progonka upper bounds
			T piLyambda = T.Pi / (_2 * T.CreateTruncating(Ny));//Ny = fftSize
			T subsupra2 = subsupra * _2;

			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
			{
				_φ[core] = new T[2 * L];
				_α[core] = new T[alfaUB + 1];
				T diag, overUnder;//a - diagonal element, b - subdiagonal,supradiagonal;SNE p.235 (37)
				T[] φ = _φ[core];
				T[] α = _α[core];
				for (int m = 0 + core; m < M; m += cCores)
				{
					calculateΦ(m, out overUnder, out diag, φ);
					progonka(m, overUnder, diag, φ, α);

					showProgress(18.0f / M);
					if (loopState.IsStopped) return;
					if (iterationsCanceled)
					{
						loopState.Stop();
						return;
					}
				}
			});

			void calculateΦ(int m, out T overUnder, out T diag, T[] φ)
			{//using m + 1 in λ(m) because m is zero based(for arrays)
				T λ2 = λDiv2(m + 1);
				T μₖ2, μₖ1;
				if (k > 16)
				{
					T sqr = T.Sqrt(λ2 * λ2 - T.One);
					T coef = T.One / (sqr * _2);
					μₖ2 = (T.Pow(λ2 + sqr, km1) - T.Pow(λ2 - sqr, km1)) * coef;//SNE p.234 bottom
					μₖ1 = (T.Pow(λ2 + sqr, k0) - T.Pow(λ2 - sqr, k0)) * coef;
				}
				else
				{
					(μₖ2, μₖ1) = UnRecursive(k - 2, λ2);//SNE p.234 bottom
				}
				T μₖ0 = λ2 * μₖ1 * _2 - μₖ2;// Un(k - 0, λ2, sqrx);
				overUnder = T.One / (μₖ1 * μₖ1 - μₖ2 * μₖ2);//SNE p.235 below (37);error - in book used μₖ0 instead of μₖ1(see p.233(34))
				diag = μₖ1 * (μₖ0 - μₖ2) * overUnder;//SNE p.235 below (37)

				for (int l = 0; l < L; l++)//SNE p.233(34)
				{
					int lp = l;//indexing p array
					int lq = l + L;//indexing q array
					T fiplus = (pq0[lq][m] + pq0[lp][m]) / ((μₖ2 - μₖ1) * _2);//SNE p.235(37)
					T fiminu = (pq0[lq][m] - pq0[lp][m]) / ((μₖ2 + μₖ1) * _2);//SNE p.235(37)
					φ[2 * l + 1 - 1] = fiplus + fiminu;//SNE p.235(37) ' - 1' for zero base
					φ[2 * l + 2 - 1] = fiplus - fiminu;//SNE p.235(37) ' - 1' for zero base
				}

				(T, T) UnRecursive(int n, T x)
				{//https://ru.wikipedia.org/wiki/%D0%9C%D0%BD%D0%BE%D0%B3%D0%BE%D1%87%D0%BB%D0%B5%D0%BD%D1%8B_%D0%A7%D0%B5%D0%B1%D1%8B%D1%88%D1%91%D0%B2%D0%B0
					if (n == 0) return (T.One, x * _2);
					if (n == 1) return (x * _2, x * x * _4 - T.One);
					T u0 = T.Zero, u1 = T.One, u2 = x * _2;
					for (int i = 0; i < n - 1; i++)
					{
						u0 = u1;
						u1 = u2;
						u2 = x * u1 * _2 - u0;
					}
					T rc1 = u2;
					u0 = u1;
					u1 = u2;
					u2 = x * u1 * _2 - u0;
					return (rc1, u2);
				}

				//[MethodImpl(MethodImplOptions.AggressiveInlining)]
				T λDiv2(int n)
				{
					T sin = T.Sin(piLyambda * T.CreateTruncating(n));
					return T.One + subsupra2 * sin * sin;//SNE p.236 under (39)
				}
			}

			void progonka(int m, T overUnder, T diag, T[] φ, T[] α)
			{
				T overDiag(int i) => ((i & 1) == 0) ? overUnder : T.One;
				T underDiag(int i) => ((i & 1) == 1) ? overUnder : T.One;
				void calcAlpha()
				{
					α[0] = overDiag(0) / diag;//SNR p.75(7)
					for (int i = 1; i <= alfaUB; i++) α[i] = overDiag(i) / (diag - underDiag(i) * α[i - 1]);//SNR p.75(7)
				}
				calcAlpha();

#if MeetingProgonka
				//ŵ as beta(left part)
				ŵ[0][m] = φ[0] / diag;//SNR p.77(12)
				for (int i = 1; i <= alfaUB; i++) ŵ[i][m] = (φ[i] + underDiag(i) * ŵ[i - 1][m]) / (diag - underDiag(i) * α[i - 1]);//SNR p.77(12,formula 2)

				//ŵ as beta(right part)
				ŵ[U][m] = φ[U] / diag;//SNR p.77(12)
				for (int i = U - 1; i > alfaUB; i--) ŵ[i][m] = (φ[i] + overDiag(i) * ŵ[i + 1][m]) / (diag - overDiag(i) * α[(U - 1) - i]);//SNR p.77(12,formula 4)

				//ŵ as beta(at right side) & ŵ as ŵ(at left side)
				ŵ[L][m] = (ŵ[alfaUB + 1][m] + α[alfaUB] * ŵ[alfaUB][m]) / (T.One - α[alfaUB] * α[alfaUB]);//SNR p.77(13,formula 3)

				for (int i = L - 1; i >= 0; i--) ŵ[i][m] += α[i] * ŵ[i + 1][m];//SNR p.77(13,formula 1)
				for (int i = L + 1; i <= U; i++) ŵ[i][m] += α[U - i] * ŵ[i - 1][m];//SNR p.77(13,formula 2)
#else
				//ŵ as beta
				ŵ[0][m] = φ[0] / diag;//SNR p.77(12)
				for (int i = 1; i <= alfaUB; i++) ŵ[i][m] = (φ[i] + underDiag(i) * ŵ[i - 1][m]) / (diag - underDiag(i) * α[i - 1]);

				for (int i = U - 1; i >= 0; i--) ŵ[i][m] += α[i] * ŵ[i + 1][m];
#endif
			}
		}

		void reversePath()
		{
			float elapsed = getExecutedSeconds(stopWatchEL, () => calculateW());//2*L
			listElapsedAdd("calculateW", elapsed);
			if (iterationsCanceled) return;

			elapsed = getExecutedSeconds(stopWatchEL, () => calculateY());
			listElapsedAdd("calculateY", elapsed);
		}

		void calculatePQVectors()
		{
			pq0 = new T[L * 2][];
			pq1 = new T[L * 2][];
			pq2 = new T[L * 2][];

			Parallel.For(0, L, optionsParallel, (l, loopState) =>
			{
				calculatePkQk(l);

				showProgress(7.0f/L);
				if (loopState.IsStopped) return;
				if (iterationsCanceled)
				{
					loopState.Stop();
					return;
				}
			});


			void calculatePkQk(int l)
			{
				int lp = l;//lower part of arrays
				pq1[lp] = new T[pqDim2];
				pq2[lp] = new T[pqDim2];
				pq0[lp] = new T[pqDim2];
				int lq = l + L;//upper part of arrays
				pq1[lq] = new T[pqDim2];
				pq2[lq] = new T[pqDim2];
				pq0[lq] = new T[pqDim2];

				int offsP = 2 * l * k;
				initCalculatePQ(lp, offsP > 1, offsP + 1);

				int offsQ = 2 * (l + 1) * k;
				initCalculatePQ(lq, offsQ < Nx - 1, offsQ);

				for (int j = 2; j <= k; j++)
				{
					calculatePQ(lp, offsP + j);//lower part of arrays
					calculatePQ(lq, offsQ + 1 - j);//upper part of arrays
				}
				//local functions
				void initCalculatePQ(int idx, bool condition, int offsFn)
				{
					Array.Clear(pq1[idx], 0, pq1[idx].Length - 2);//can not call;if call - no need set additional elements
					if (fKsi == null && condition) vectorAssignShortRow(pq2[idx], fn[offsFn - 1]);
					else vectorAssignRow(pq2[idx], fn[offsFn - 1]);
				}

				void vectorAssignRow(T[] dst, T[] src)
				{
					for (int i = 0; i < M; i++) dst[i] = src[i + 1];
				}

				void vectorAssignShortRow(T[] dst, T[] src)
				{
					dst[0] = src[1];
					dst[M - 1] = src[M];
				}

				void calculatePQ(int ind, int row)
				{
					T[] tmp = pq0[ind];
					pq0[ind] = pq1[ind];
					pq1[ind] = pq2[ind];
					pq2[ind] = tmp;
					matrixCMultiplyVector(pq2[ind], pq1[ind]);
					vectorSubtractVector(pq2[ind], pq0[ind]);
					if(fKsi == null) vectorAddShortRow(pq2[ind], fn[row - 1]);
					else vectorAddRow(pq2[ind], fn[row - 1]);
				}

				void matrixCMultiplyVector(T[] dst, T[] src)
				{
					dst[0] = cBase * src[0] - bi * src[1];
					for (int i = 1; i < M - 1; i++) dst[i] = -ai * src[i - 1] + cBase * src[i] - bi * src[i + 1];
					dst[M - 1] = -ai * src[M - 2] + cBase * src[M - 1];
				}

				void vectorSubtractVector(T[] dst, T[] src)
				{
					for (int i = 0; i < M; i++) dst[i] -= src[i];
				}
			}

			void vectorAddRow(T[] dst, T[] src)
			{
				for (int i = 0; i < M; i++) dst[i] += src[i + 1];
			}

			void vectorAddShortRow(T[] dst, T[] src)
			{
				dst[0] += src[1];
				dst[M - 1] += src[M];
			}
		}

		void calculateĜ()
		{//memory used while working this func is 6L*(M+2), pq0 is ĝ
			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
			{
				for (int l = 0 + core; l <= L - 1; l += cCores)
				{//'-1' for zero based
					int lp = l;//lower parts of arrays
					int lq = l + L;//upper parts of arrays
					fftM.calculate(core, (m) => pq1[lq][m - 1] - pq2[lp][m - 1], (m, val) => pq0[lp][m - 1] = normFactor * val);//SNE p.232 (31), p.235 (38)
					fftM.calculate(core, (m) => pq1[lp][m - 1] - pq2[lq][m - 1], (m, val) => pq0[lq][m - 1] = normFactor * val);//SNE p.232 (31), p.235 (38)

					showProgress(13.0f / L);
					if (loopState.IsStopped) return;
					if (iterationsCanceled)
					{
						loopState.Stop();
						return;
					}
				}
			});
			pq2 = null;
			GC.Collect();
		}

		void calculateW()
		{//memory used while working this func is 4L*(M+2);
			int rem = L > 4 ? L / 4 : 1;//8 pictures(loop 2L)
			un = new T[Nx + 1][];
			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
			{
				for (int iW = 0 + core; iW < 2 * L; iW += cCores)
				{
					int iUN = iW * k + ((iW & 1) != 1 ? 1 : k);
					int iPQ = (iW >> 1) + L * (iW & 1);//lower or upper part of array
					un[iUN] = pq1[iPQ];//instead of new double[Ny + 1];
					fftM.calculate(core, (m) => ŵ[iW][m - 1], (j, val) => un[iUN][j] = normFactor * val);//SNE p.236, (39)
					if (unShow != null) for (int j = 1; j < Ny; j++) unShow[iUN][j] = float.CreateTruncating(un[iUN][j]);
					if (iW % rem == 0) UtilsPict.addPicture(lstBitmap, true, null/*minMax*/, new Adapter2D<float>(Nx + 1, Ny + 1, (m, k) => unShow[m][k]), fCreateBitmap);

					showProgress(11.0f / L);
					if (loopState.IsStopped) return;
					if (iterationsCanceled)
					{
						loopState.Stop();
						return;
					}
				}
			});
			ŵ = null;//but p0,q0(which is used fot it) are not nulling
		}

		void calculateY()
		{
			int rem = L > 8 ? L / 8 : 1;//8 pictures
			object locker = (lstBitmap != null) ? new object() : null;
			int val = -1;
			Parallel.For(0, L, optionsParallel, (ll, loopState) =>
			{
				int l = ll;
				if (lstBitmap != null) lock (locker)
					{
						val++;
						l = val;
					}
				calculateYFromLeft(l);//L*(k-1)
				if (l % rem == 0) UtilsPict.addPicture(lstBitmap, true, null/*minMax*/, new Adapter2D<float>(Nx + 1, Ny + 1, (i, j) => unShow[i][j]), fCreateBitmap);
				calculateYFromRight(l);//L*k-1)
				if (l % rem == 0) UtilsPict.addPicture(lstBitmap, true, null/*minMax*/, new Adapter2D<float>(Nx + 1, Ny + 1, (i, j) => unShow[i][j]), fCreateBitmap);

				showProgress(40.0f / L);
				if (loopState.IsStopped) return;
				if (iterationsCanceled)
				{
					loopState.Stop();
					return;
				}
			});
			//local functions
			void calculateYFromLeft(int l)//SNE p.232(30)
			{
				int idxLeft = 2 * l * k + 2;
				for (int i = idxLeft; i <= (2 * l + 1) * k; i++)
				{
					un[i] = (i != idxLeft) ? fn[i - 3] : pq0[l];//instead of new double[Ny + 1]
					matrixCMultiplyArrayVector(un, i, i - 1);
					if (i - 2 > 0) arrayRowSubtractRow(un, i, i - 2);//skip subtracting left edge column which assumed to be zero
					if ((fKsi == null) && i - 1 > 1) arrayRowSubtractShortVector(un, i, fn[i - 2]);
					else arrayRowSubtractVector(un, i, fn[i - 2]);

					if (unShow != null) for (int j = 1; j < Ny; j++) unShow[i][j] = float.CreateTruncating(un[i][j]);
				}
			}

			void calculateYFromRight(int l)//SNE p.232(30)
			{
				int idxRight = 2 * (l + 1) * k - 1;
				for (int i = idxRight; i >= (2 * l + 1) * k + 1; i--)
				{
					un[i] = (i != idxRight) ? fn[i + 1] : pq0[l + L];//instead of new double[Ny + 1]
					matrixCMultiplyArrayVector(un, i, i + 1);
					if (i + 2 < Nx) arrayRowSubtractRow(un, i, i + 2);//skip subtracting right edge column which assumed to be zero
					if ((fKsi == null) && i + 1 < Nx - 1) arrayRowSubtractShortVector(un, i, fn[i]);
					else arrayRowSubtractVector(un, i, fn[i]);

					if (unShow != null) for (int j = 1; j < Ny; j++) unShow[i][j] = float.CreateTruncating(un[i][j]);
				}
			}

			void matrixCMultiplyArrayVector(T[][] dst, int rowDst, int rowSrc)
			{
				dst[rowDst][1] = cBase * dst[rowSrc][1] - bi * dst[rowSrc][2];
				for (int i = 2; i < M; i++)
				{
					dst[rowDst][i] = -ai * dst[rowSrc][i - 1] + cBase * dst[rowSrc][i] - bi * dst[rowSrc][i + 1];
				}
				dst[rowDst][M] = -ai * dst[rowSrc][M - 1] + cBase * dst[rowSrc][M];
			}

			void arrayRowSubtractRow(T[][] dst, int rowDst, int rowSrc)
			{
				for (int j = 1; j <= M; j++) dst[rowDst][j] -= dst[rowSrc][j];
			}

			void arrayRowSubtractVector(T[][] dst, int rowDst, T[] src)
			{
				for (int j = 1; j <= M; j++) dst[rowDst][j] -= src[j];
			}

			void arrayRowSubtractShortVector(T[][] dst, int rowRes, T[] src)
			{
				dst[rowRes][1] -= src[1];
				dst[rowRes][M] -= src[M];
			}
		}

		void restoreBounds(T[][] uu)
		{
			Parallel.For(1, Nx, optionsParallel, (i) =>
			{
				uu[i][0] = bndB[i - 1];
				uu[i][Ny] = bndT[i - 1];
			});

			uu[0] = new T[Ny + 1];
			uu[Nx] = new T[Ny + 1];
			Parallel.For(1, Ny, optionsParallel, (j) =>
			{
				uu[0][j] = bndL[j - 1];
				uu[Nx][j] = bndR[j - 1];
			});

			T nan = T.Zero / T.Zero;
			uu[0][0] = nan;
			uu[Nx][0] = nan;
			uu[0][Ny] = nan;
			uu[Nx][Ny] = nan;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			if (unShow != null)
			{
				GridIterator.iterateEdgesAndFillInternalPoints(Nx + 1, Ny + 1, null, (i, j) => unShow[i][j] = float.NaN);
				fillBounds(unShow);

				void fillBounds(float[][] uu)
				{
					Parallel.For(1, Nx, optionsParallel, (i) =>
					{
						uu[i][0] = float.CreateTruncating(bndB[i - 1]);
						uu[i][Ny] = float.CreateTruncating(bndT[i - 1]);
					});

					Parallel.For(1, Ny, optionsParallel, (j) =>
					{
						uu[0][j] = float.CreateTruncating(bndL[j - 1]);
						uu[Nx][j] = float.CreateTruncating(bndR[j - 1]);
					});

					uu[0][0] = (uu[1][0] + uu[0][1]) / 2;
					uu[Nx][0] = (uu[Nx - 1][0] + uu[Nx][1]) / 2;
					uu[0][Ny] = (uu[1][Ny] + uu[0][Ny - 1]) / 2;
					uu[Nx][Ny] = (uu[Nx - 1][Ny] + uu[Nx][Ny - 1]) / 2;
				}
			}
		}

		public void cancelIterations() { iterationsCanceled = true; }
		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }

		public override void cleanup()
		{
			fn = null;
			unShow = null;
			pq0 = null;
			pq1 = null;
			pq2 = null;
			ŵ = null;
			fftM = null;
			base.cleanup();
		}

		void showProgress(float count)
		{
			curProgress += count;
			if (curProgress - oldProgress > 0.99)
			{
				reportProgress((int)curProgress);
				oldProgress = curProgress;
			}
		}
		override public string getElapsedInfo() { return timesElapsed(); }
	}
}

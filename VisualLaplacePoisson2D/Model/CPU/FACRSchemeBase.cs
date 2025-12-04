using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class FACRSchemeBase<T> : DirectJagged2Scheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		protected interface IProgonka
		{
			int calcAlpha(T diagElem, int idxAlfa);
			void progonkaWithFunction(int idxDelta, int idxAlfa, int rowcolRes, int k);
			void progonkaWithCoeff(T coeff, int idxAlfa, int rowRes, int k);
		}
		protected int M, L, ML;
		protected ParallelOptions optionsParallel;
		protected List<BitmapSource> lstBitmap;
		protected Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;
		protected float[][] unShow;
		protected Action<double> reportProgress;
		protected int progressSteps, curProgress;
		protected T stepX2, stepY2, stepX, stepY;
		protected Func<T, T, T> funcKsi;
		protected FFTCalculator<T> fft;
		protected bool iterationsCanceled;
		protected int cCores;
		protected Action<int> addPictureAction = null;
		protected int rem1, rem24, rem3, rem5, remPict;//rem - remainder
		protected IProgonka progonka;
		T _2 = T.CreateTruncating(2);

		public FACRSchemeBase(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsi, int paramL, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(cXSegments + 1, cYSegments + 1, fKsi == null)
		{
			stepX = stepXIn;
			stepY = stepYIn;
			stepX2 = stepXIn * stepXIn;
			stepY2 = stepYIn * stepYIn;
			funcKsi = fKsi;
			reportProgress = reportProgressIn;
			this.cCores = cCores;
			optionsParallel = new ParallelOptions() { MaxDegreeOfParallelism = cCores };
			L = paramL;

			this.lstBitmap = lstBitmap;
			this.fCreateBitmap = fCreateBitmap;
			if (lstBitmap != null)
			{
				unShow = new float[cXSegments + 1][];
				for (int i = 0; i < cXSegments + 1; i++) unShow[i] = new float[cYSegments + 1];
			}

			if (unShow != null) addPictureAction = (j) =>
			{
				fillUnShowForFFT(j);
				if ((remPict > 0) && (j % remPict == 0)) UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (i, k) => unShow[i][k]), fCreateBitmap);
			};

			progressSteps = 100;//see rem1,2,etc
			curProgress = 0;
		}

		protected void FFTCalculate(int N, Func<int, int, T> input, Action<int, int, T> act, Action<int> addPictureAction)
		{
			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
			{
				if (loopState.IsStopped) return;
				for (int j = core + 1; j <= N - 1; j += cCores)//1 ≤ j ≤ N - 1
				{
					fft.calculate(core, (i) => input(i, j), (i, val) => act(i, j, val));

					addPictureAction?.Invoke(j);

					if ((rem24 > 0) && (j % rem24 == 0)) showProgress();

					if (iterationsCanceled)
					{
						loopState.Stop();
						return;
					}
				}
			});
		}

		protected void reductionLSteps(int N, T hXY2)
		{
			if (L == 0) return;
			int m = M;
			T diagElem = (T.One + hXY2) * _2;

			T[] diag = new T[1] { T.Zero };
			int cMatrices;

			T[][] multiplied, accum;
			multiplied = new T[cCores][];
			accum = new T[cCores][];
			for (int i = 0; i < cCores; i++)
			{
				multiplied[i] = new T[N - 1];
				accum[i] = new T[N - 1];
			}

			void matricesCMultipleVector(int core, int row)
			{
				initMultiplied(multiplied[core], row);
				for (int k = 0; k < cMatrices; k++)
				{
					accum[core][0] = diag[k] * multiplied[core][0] - hXY2 * (multiplied[core][1]);
					for (int j = 1; j <= N - 3; j++)
					{
						accum[core][j] = diag[k] * multiplied[core][j] - hXY2 * (multiplied[core][j - 1] + multiplied[core][j + 1]);
					}
					accum[core][N - 2] = diag[k] * multiplied[core][N - 2] - hXY2 * (multiplied[core][N - 3]);
					UtilsSwap.swap(ref multiplied[core], ref accum[core]);
				}
			}
			for (int l = 1; l <= L; l++)
			{
				diag = new T[1 << (l - 1)];
				generateSqrtCoefs<T>(l - 1, (i, val) => diag[i] = (diagElem + val));
				cMatrices = diag.GetUpperBound(0) + 1;
				int n = 1 << (l - 1);
				Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
				{
					if (loopState.IsStopped) return;
					for (int i1 = core + 1; i1 < m; i1 += cCores)
					{
						int i = i1 << l;
						matricesCMultipleVector(core, i);
						doReduction(i, n, multiplied[core]);

						if ((rem1 > 0) && (i1 % rem1 == 0)) showProgress();

						if (iterationsCanceled)
						{
							loopState.Stop();
							return;
						}
					}
				});
				m >>= 1;
			}
		}

		protected void progonkaEvenVectors(T step2, T step2DivStep2, int N)
		{
			int[] matrixOrder = UtilsChebysh.reductionParams(L);//no need ?!
			T[][] diagElems = new T[cCores][];
			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
			{
				diagElems[core] = new T[1 << L];
				if (loopState.IsStopped) return;
				for (int k2 = core + 1; k2 < ML; k2 += cCores)
				{
					int kk = k2 << L;//even index

					generateProgonkaEvenCoeffs(diagElems[core], k2, L, step2DivStep2, N);
					for (int i = 0; i < 1 << L; i++)//'1 << L' == count of diagElems
					{
						int kUp = progonka.calcAlpha(diagElems[core][matrixOrder[i] - 1], core);
						progonka.progonkaWithCoeff((i == 0) ? step2 : step2DivStep2, core, kk, kUp);
					}

					if ((rem3 > 0) && (k2 % rem3 == 0)) showProgress();
					if (iterationsCanceled)
					{
						loopState.Stop();
						return;
					}
				}
			});
		}

		protected void reverseSteps(int N, T step2DivStep2)
		{
			if (L == 0) return;
			int remPict2 = M / 10;
			T[] coefs;
			int[] kUp; 
			for (int l = L; l >= 1; l--)
			{
				coefs = new T[1 << (l - 1)];
				generateSqrtCoefs<T>(l - 1, (i, val) => coefs[i] = (val + _2));
				kUp = new int[1 << (l - 1)];
				for (int i = 0; i < coefs.Length; i++) kUp[i] = progonka.calcAlpha(step2DivStep2 * coefs[i] + _2, i);
				int mL = N >> l;
				int idxDelta = 1 << (l - 1);
				Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
				{
					if (loopState.IsStopped) return;
					for (int iLoop = core + 1; iLoop <= mL; iLoop += cCores)
					{
						int il = (iLoop << l) - idxDelta;//2,6,10,14,...(for L==2)

						progonka.progonkaWithFunction(idxDelta, 0, il, kUp[0]);
						for (int i = 1; i < 1 << (l - 1); i++) progonka.progonkaWithCoeff(step2DivStep2, i, il, kUp[i]);

						if (unShow != null)
						{
							fillUnShowForReverseSteps(il);
							if ((remPict2 > 0) && (iLoop % remPict2 == 0)) UtilsPict.addPicture(lstBitmap, true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (i, k) => unShow[i][k]), fCreateBitmap);
						}

						if ((rem5 > 0) && (iLoop % rem5 == 0)) showProgress();
						if (iterationsCanceled)
						{
							loopState.Stop();
							return;
						}
					}
				});
			}
		}

		virtual protected void fillUnShowForReverseSteps(int il)
		{
		}

		virtual protected void fillUnShowForFFT(int idx)
		{
		}

		virtual protected void initMultiplied(T[] multiplied, int rowOrCol)
		{
		}

		virtual protected void doReduction(int i, int n, T[] multiplied)
		{
		}

		protected void showProgress()
		{
			curProgress++;
			reportProgress(curProgress * 100.0 / progressSteps);
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			if (unShow != null) GridIterator.iterateEdgesAndFillInternalPoints(N1 + 1, N2 + 1, (i, j) => unShow[i][j] = float.CreateTruncating(un[i][j]), (i, j) => unShow[i][j] = float.NaN);
		}

		protected void initRigthHandSide()
		{//[SNR] p.199, above (2)
			if (funcKsi != null) GridIterator.iterate(N1, N2, (i, j) => un[i][j] = funcKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
		}

		protected void transferBoundaryValuesToNearBoundaryNodes()
		{//[SNR] p.190, (19)
			Parallel.For(1, N2, (j) =>
			{
				un[1][j] += un[0][j] / stepX2;
				un[N1 - 1][j] += un[N1][j] / stepX2;
			});
			Parallel.For(1, N1, (i) =>
			{
				un[i][1] += un[i][0] / stepY2;
				un[i][N2 - 1] += un[i][N2] / stepY2;
			});
		}

		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public override void cleanup()
		{
			base.cleanup();
		}

		static void generateProgonkaEvenCoeffs(T[] diagCoeffs, int k2, int L, T step2DivStep2, int N)
			//, IAdditionOperators<T, double, T>, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>, IDivisionOperators<T, double, T>
		{
			//replace C with 'C²-2' for next l param
			//C - 2cos(x) порождает(C - 2cos(x)) * (C + 2cos(x))
			//C + 2cos(x) порождает(C - 2sin(x)) * (C + 2sin(x))
			//C - 2sin(x) порождает(C - 2 * cos(x - π / 4)) * (C + 2 * cos(x - π / 4))
			//C + 2sin(x) порождает(C - 2 * sin(x - π / 4)) * (C + 2 * sin(x - π / 4))
			T _2 = T.CreateTruncating(2);
			T _4 = T.CreateTruncating(4);
			T _k2 = T.CreateTruncating(k2);

			T step2DivStep2Doubled = step2DivStep2 * _2;
			T piDivN = T.Pi / T.CreateTruncating(N);
			T ck2base = step2DivStep2Doubled + _2;//[SNR] p.202, (21)
			T diagDelta1 = T.Zero;
			if (L >= 0)
			{
				diagDelta1 = step2DivStep2Doubled * T.Cos(piDivN * _k2);//there are equivalent formulas in progonkaCalculate() in VariablesSeparationSchemeProgonka
				diagCoeffs[0] = ck2base - diagDelta1;
			}
			if (L >= 1)
			{
				diagCoeffs[1] = ck2base + diagDelta1;
			}
			if (L >= 2)
			{
				T diagDelta2 = step2DivStep2Doubled * T.Sin(piDivN * _k2);

				diagCoeffs[2] = ck2base - diagDelta2;
				diagCoeffs[3] = ck2base + diagDelta2;
			}
			if (L >= 3)
			{
				T diagDelta3 = step2DivStep2Doubled * T.Cos(piDivN * _k2 - T.Pi / _4);
				T diagDelta4 = step2DivStep2Doubled * T.Sin(piDivN * _k2 - T.Pi / _4);
				//T diagDelta3 = step2DivStep2Doubled * Sin(k2 * piDivN);
				//T diagDelta4 = step2DivStep2Doubled * Cos(k2 * piDivN - PI);

				diagCoeffs[4] = ck2base - diagDelta3;
				diagCoeffs[5] = ck2base + diagDelta3;
				diagCoeffs[6] = ck2base - diagDelta4;
				diagCoeffs[7] = ck2base + diagDelta4;
			}
			if (L >= 4)
			{//(C-2*cos(x-π/4))*(C+2*cos(x-π/4))*(C-2*sin(x-π/4))*(C+2*sin(x-π/4))*(C-2*cos(x-π/2))*(C+2*cos(x-π/2))*(C-2*sin(x-π/2))*(C + 2 * sin(x - π / 2))
				/*T diagDelta5 = step2DivStep2Doubled * Cos(k2 * piDivN - PI / 4.0);
				T diagDelta6 = step2DivStep2Doubled * Sin(k2 * piDivN - PI / 4.0);
				T diagDelta7 = step2DivStep2Doubled * Cos(k2 * piDivN - PI / 2.0);
				T diagDelta8 = step2DivStep2Doubled * Sin(k2 * piDivN - PI / 2.0);

				diagCoeffs[ 8] = ck2base - diagDelta5;
				diagCoeffs[ 9] = ck2base + diagDelta5;
				diagCoeffs[10] = ck2base - diagDelta6;
				diagCoeffs[11] = ck2base + diagDelta6;
				diagCoeffs[12] = ck2base - diagDelta7;
				diagCoeffs[13] = ck2base + diagDelta7;
				diagCoeffs[14] = ck2base - diagDelta8;
				diagCoeffs[15] = ck2base + diagDelta8;*/
				int cLoop = L - 4 + 1;
				int cAdd = 8;
				for (int i = 0; i < cLoop; i++)
				{
					for (int j = 0; j < cAdd; j++) diagCoeffs[j + cAdd] = diagCoeffs[j];
					cAdd <<= 1;
				}
			}
		}
	}
}
/*
C⁽¹⁾=[C]²-2E (3.35); C⁽¹⁾-2cos *2* (3.45)-(3.46)
C⁽²⁾=[C⁽¹⁾]²-2E=[C²-2E]²-2E (3.60); C⁽²⁾-2cos *4* (3.65)
C⁽³⁾=[C⁽²⁾]²-2E=([C⁽¹⁾]²-2E)²-2E=((C²-2E)²-2E)²-2E (3.78); C⁽³⁾-2cos *8* (?.??)
C⁽⁴⁾=[C⁽³⁾]²-2E=...; C⁽⁴⁾-2cos *16* (?.??)
C⁽ˡ⁾=[C⁽ˡ⁻¹⁾]²-2E (3.84)
*/

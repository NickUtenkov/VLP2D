using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class VariablesSeparationSchemeNoProgonka<T> : VariablesSeparationScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T[][] fik2i, fik1k2;
		T[] lyambda1, lyambda2;
		FFTCalculator<T> fftN1;

		public VariablesSeparationSchemeNoProgonka(int cXSegments, int cYSegments, T stepX, T stepY, int cCores, Func<T, T, T> fKsi, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, stepX, stepY, cCores, fKsi, lstBitmap0, fCreateBitmap, reportProgressIn)
		{
			fik2i = un;
			fik1k2 = un;

			fftN1 = fftN2;
			if (N1 != N2) fftN1 = new FFTCalculator<T>(cCores, N1);

			progressSteps = 100;//4 loops by 25
			rem1 = N1 / 15;//twice 15(in calculate) plus 70(rem2)
			remPict = N1 / 10;
		}

		override public T doIteration(int iter)
		{
			initElapsedList();

			float elapsed = getExecutedSeconds(stopWatchEL, () => { calcLyambdas(); initRigthHandSide(fKsi, stepX, stepY); transferBoundaryValuesToNearBoundaryNodes(); });
			listElapsedAdd("lyambdas, RHS, transfer", elapsed);

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2Calculate(T.One, null));//[SNR] p.192, (24), using fn[i, k] & fik2i[i, k]
			listElapsedAdd("FFTN2_1", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN1Calculate());
			listElapsedAdd("FFTN1", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2Calculate(T.CreateTruncating(4.0 / (N1 * N2)), addPictureAction));//[SNR] p.192, (27), using uk2i[i, k] & fn[i, k]
			listElapsedAdd("FFTN2_2", elapsed);

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		public override string getElapsedInfo() { return timesElapsed(); }

		public override void cleanup()
		{
			fik2i = null;
			fik1k2 = null;
			fftN1 = null;
			lyambda1 = null;
			lyambda2 = null;
			base.cleanup();
		}

		void calcLyambdas()
		{
			lyambda1 = calcLyambda(N1, stepX2);
			lyambda2 = ((N2 != N1) || T.Abs(stepX2 - stepY2) > T.CreateTruncating(1E-10)) ? calcLyambda(N2, stepY2) : lyambda1;
		}

		void fftN1Calculate()
		{
			int rem2 = N2 / 70;
			Parallel.For(0, cCores, optionsParallel, (core, loopState) =>
			{
				if (loopState.IsStopped) return;
				for (int k2 = 1 + core; k2 < N2; k2 += cCores)
				{
					fftN1.calculate2(core, (k) => fik2i[k][k2], (k, val) => val / (lyambda1[k] + lyambda2[k2]), (k1, val) => fik1k2[k1][k2] = val);//[SNR] p.192, (25),p.192, (26); fik1k2 already divided by lyambdak1k2(see above)

					if ((rem2 > 0) && (k2 % rem2 == 0)) showProgress();
					if (areIterationsCanceled())
					{
						loopState.Stop();
						return;
					}
				}
			});
		}

		T[] calcLyambda(int n, T step2)
		{
			T[] lyambda = new T[n];
			T pi2N = T.Pi / T.CreateTruncating(n) / T.CreateTruncating(2);
			for (int i = 1; i < n; i++)
			{
				T sin = T.Sin(pi2N * T.CreateTruncating(i));
				lyambda[i] = sin * sin * T.CreateTruncating(4.0) / step2;
			}

			return lyambda;
		}
	}
}


using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static System.Math;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class FACRScheme<T> : FACRSchemeBase<T>, IScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		protected T y2DivX2;
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());

		public FACRScheme(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsi, int paramL, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn)
			: base(cXSegments, cYSegments, stepXIn, stepYIn, cCores, fKsi, paramL, lstBitmap0, fCreateBitmap, reportProgressIn)
		{
			y2DivX2 = stepY2 / stepX2;

			M = N1 >> 1;
			ML = N1 >> L;

			remPict = N2 / 10;
			rem1 = M / 10;
			rem24 = N2 / 30;//2 & 4 steps
			rem3 = M / 20;
			rem5 = M / 10;

			fft = new FFTCalculator<T>(cCores, ML);
			progonka = new Progonka(un, stepY2, y2DivX2, cCores, paramL, αCC);
		}

		public T doIteration(int iter)
		{
			initElapsedList();

			float elapsed = getExecutedSeconds(stopWatchEL, () => { initRigthHandSide(); transferBoundaryValuesToNearBoundaryNodes(); });
			listElapsedAdd("initRHS", elapsed);
			try
			{
				elapsed = getExecutedSeconds(stopWatchEL, () => reductionLSteps(N2, stepX2 / stepY2));//[SNR] p.202 (19)
				listElapsedAdd("reductionLSteps", elapsed);

				elapsed = getExecutedSeconds(stopWatchEL, () => FFTCalculate(N2, (i, j) => un[i << L][j], (i, j, val) => un[i << L][j] = val, null));//[SNR] p.202 (20)
				listElapsedAdd("FFT1", elapsed);
				if (iterationsCanceled) return T.Zero;

				elapsed = getExecutedSeconds(stopWatchEL, () => progonkaEvenVectors(stepY2, y2DivX2, N1));//[SNR] p.202 (22)
				listElapsedAdd("progonkaEvenVectors", elapsed);
				if (iterationsCanceled) return T.Zero;

				T coef = T.CreateTruncating(2.0 * (1 << L)) / T.CreateTruncating(N1);
				elapsed = getExecutedSeconds(stopWatchEL, () => FFTCalculate(N2, (i, j) => un[i << L][j], (i, j, val) => un[i << L][j] = coef * val, addPictureAction));//[SNR] p.203 (23)
				listElapsedAdd("FFT2", elapsed);
				if (iterationsCanceled) return T.Zero;

				elapsed = getExecutedSeconds(stopWatchEL, () => reverseSteps(N1, y2DivX2));//[SNR] p.202 (24)
				listElapsedAdd("reverseSteps", elapsed);
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.Message, ex.TargetSite.Name);
			}

			return T.Zero;
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		override protected void initMultiplied(T[] multiplied, int row)
		{
			for (int j = 0; j <= N2 - 2; j++) multiplied[j] = un[row][j + 1];
		}

		override protected void doReduction(int i, int n, T[] multiplied)
		{
			T prevVal = un[i - n][1] + un[i + n][1] + multiplied[0];//hXY2 * (0 + un[i, 2])
			for (int j = 2; j < N2 - 1; j++)
			{
				T curVal = un[i - n][j] + un[i + n][j] + multiplied[j - 1];//[Zor] (3.38,3.59, 3.71);[SNR] p.202 (19) and p.201 (15)
				un[i][j - 1] = prevVal;
				prevVal = curVal;
			}
			un[i][N2 - 2] = prevVal;
			un[i][N2 - 1] = un[i - n][N2 - 1] + un[i + n][N2 - 1] + multiplied[N2 - 2];//hXY2 * (un[i, N2 - 2] + 0)
		}

		override protected void fillUnShowForReverseSteps(int il)
		{
			for (int j = 1; j < N2; j++) unShow[il][j] = float.CreateTruncating(un[il][j]);
		}

		override protected void fillUnShowForFFT(int idx)
		{
			for (int i = 1; i < ML; i++) unShow[i << L][idx] = float.CreateTruncating(un[i << L][idx]);
		}

		protected class Progonka : IProgonka
		{
			T[][] un;
			int N1, N2;
			T stepY2, y2DivX2;
			T[][] alfaJagged;
			readonly int alfaUpperBound, progonkaUpperBound;
			AlfaСonvergentUpperBoundEpsilon αCC;

			public Progonka(T[][] un, T stepY2, T y2DivX2, int cCores, int paramL, AlfaСonvergentUpperBoundEpsilon αCC)
			{
				this.un = un;
				this.stepY2 = stepY2;
				this.y2DivX2 = y2DivX2;
				this.αCC = αCC;

				N1 = un.GetUpperBound(0);
				N2 = un[0].GetUpperBound(0);

				int nAlfa = Max(cCores, (paramL > 0) ? 1 << (paramL - 1) : 0);//cCores in progonkaEvenVectors, 2^(paramL - 1) in reverseSteps
				alfaJagged = new T[nAlfa][];
				for (int i = 0; i < nAlfa; i++) alfaJagged[i] = new T[N2 - 1];
				alfaUpperBound = N2 - 2;
				progonkaUpperBound = N2 - 1;
			}

			public int calcAlpha(T diagElem, int idxAlfa)
			{
				int k = αCC.upperBound(diagElem, alfaUpperBound);
				T[] alfa = alfaJagged[idxAlfa];
				alfa[0] = T.One / diagElem;//[SNR] p.75(7)
				for (int i = 1; i <= k; i++) alfa[i] = T.One / (diagElem - alfa[i - 1]);//[SNR] p.75(7)
				return k;
			}

			public void progonkaWithFunction(int idxDelta, int idxAlfa, int rowRes, int k)
			{//beta array is not used - it is placed to un
			 //using 0 instead of un[0, colRes] which is NOT zero(boundary value), but assumed to be zero(transferred value to near boundary node)
				T[] alfa = alfaJagged[idxAlfa];
				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				int ind(int idx) => (idx < k) ? idx : k;
				T uSum = ((rowRes - idxDelta != 0) ? un[rowRes - idxDelta][1] : T.Zero) + ((rowRes + idxDelta != N1) ? un[rowRes + idxDelta][1] : T.Zero);
				T rhs1 = stepY2 * un[rowRes][1] + y2DivX2 * uSum;//[Zor] p.131 (3.72)

				un[rowRes][1] = (rhs1) * alfa[ind(0)];
				for (int j = 2; j <= progonkaUpperBound; j++)
				{
					uSum = ((rowRes - idxDelta != 0) ? un[rowRes - idxDelta][j] : T.Zero) + ((rowRes + idxDelta != N1) ? un[rowRes + idxDelta][j] : T.Zero);
					rhs1 = stepY2 * un[rowRes][j] + y2DivX2 * uSum;//[Zor] p.131 (3.72)
					un[rowRes][j] = (rhs1 + un[rowRes][j - 1]) * alfa[ind(j - 1)];//[SNR] p.75(7);
				}

				//using 0 instead of un[N2, colRes] which is NOT zero(boundary value), but assumed to be zero(transferred value to near boundary node)
				//not adding zero value for un[N2 - 1, colRes] += alf(N2 - 2) * 0;
				for (int j = progonkaUpperBound - 1; j >= 1; j--) un[rowRes][j] += alfa[ind(j - 1)] * un[rowRes][j + 1];//[SNR] p.75(7);
			}

			public void progonkaWithCoeff(T coeff, int idxAlfa, int rowRes, int k)
			{//beta array is not used - it is placed to un
			 //using 0 instead of un[0, colRes] which is NOT zero(boundary value), but assumed to be zero(transferred value to near boundary node)
				T[] alfa = alfaJagged[idxAlfa];
				[MethodImpl(MethodImplOptions.AggressiveInlining)]
				int ind(int idx) => (idx < k) ? idx : k;
				un[rowRes][1] = (coeff * un[rowRes][1]) * alfa[ind(0)];
				for (int j = 2; j <= progonkaUpperBound; j++) un[rowRes][j] = (coeff * un[rowRes][j] + un[rowRes][j - 1]) * alfa[ind(j - 1)];//[SNR] p.75(7);

				//using 0 instead of un[N2, colRes] which is NOT zero(boundary value), but assumed to be zero(transferred value to near boundary node)
				//not adding zero value for un[N2 - 1, colRes] += alf(N2 - 2) * 0;
				for (int j = progonkaUpperBound - 1; j >= 1; j--) un[rowRes][j] += alfa[ind(j - 1)] * un[rowRes][j + 1];//[SNR] p.75(7);
			}
		}

		public override void cleanup()
		{
			fft = null;
			progonka = null;
			base.cleanup();
		}
	}
}

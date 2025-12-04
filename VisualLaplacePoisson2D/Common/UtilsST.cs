using System;
using System.Numerics;

namespace VLP2D.Common
{
	internal class UtilsST
	{
		public static Tuple<T, T> sinTransformCoeffs<T>(int n) where T : INumberBase<T>, ITrigonometricFunctions<T>//, IMultiplyOperators<T, double, T>, IDivisionOperators<T, double, T>
		{
			T theta = T.Pi / T.CreateTruncating(n);
			T sinHalfTheta = T.Sin(theta * T.CreateTruncating(0.5));
			return new Tuple<T, T>(sinHalfTheta * sinHalfTheta * T.CreateTruncating(-2.0), T.Sin(theta));
		}

		//Clive Temperton On the FACR(l) algorithm for the discrete poisson equation.pdf, APPENDIX: AN ALGORITHM FOR THE SINE TRANSFORM p.15-16
		//code from numerical_recipes3.pdf p.647(call to realft replaced with FFTW)
		public static void sinTransformPreprocess<T>(Span<T> y, int rowOffset, int batches, int cols, int n, T wpr, T wpi) where T : INumber<T>//, IMultiplyOperators<T, double, T>
		{
			T y1, y2, wtemp;
			int offs = rowOffset;
			T _05 = T.CreateTruncating(0.5);

			for (int k = 0; k < batches; k++)
			{
				T wi = T.Zero, wr = T.One;

				y[offs + 0] = T.Zero;
				for (int j = 1; j < (n >> 1) + 1; j++)
				{
					wr = (wtemp = wr) * wpr - wi * wpi + wr;
					wi = wi * wpr + wtemp * wpi + wi;
					y1 = wi * (y[offs + j] + y[offs + n - j]);
					y2 = (y[offs + j] - y[offs + n - j]) * _05;
					y[offs + j + 0] = y1 + y2;
					y[offs + n - j] = y1 - y2;
				}
				offs += cols;//== dist
			}
		}

		public static void sinTransformPreprocessVertical(double[] y, int colOffset, int batches, int cols, int n, double wprInitial, double wpiInitial)
		{
			double y1, y2, wtemp;
			int offs = colOffset;

			for (int k = 0; k < batches; k++)
			{
				double wi = 0.0, wr = 1.0;
				double wpr = wprInitial;
				double wpi = wpiInitial;

				y[offs + 0] = 0.0;
				for (int j = 1; j < (n >> 1) + 1; j++)
				{
					wr = (wtemp = wr) * wpr - wi * wpi + wr;
					wi = wi * wpr + wtemp * wpi + wi;
					y1 = wi * (y[offs + j * cols] + y[offs + (n - j) * cols]);
					y2 = 0.5 * (y[offs + j * cols] - y[offs + (n - j) * cols]);
					y[offs + (j + 0) * cols] = y1 + y2;
					y[offs + (n - j) * cols] = y1 - y2;
				}
				offs += 1;//== dist
			}
		}

		public static void sinTransformPostprocessR2C<T>(Span<T> y, int offs, int batches, int batchSize, int n) where T : INumberBase<T>//IUnaryNegationOperators<T, T>, IAdditionOperators<T, T, T>, IMultiplyOperators<T, double, T>
		{
			bool fftSizeOdd = (n & 1) == 1;
			T _05 = T.CreateTruncating(0.5);
			//{R0, 0} {R1, I1} {R2, I2} ... {R(n+1)/2-1, I(n+1)/2-1} {Rn/2, 0} after r2c FFT, fftSize == n
			for (int k = 0; k < batches; k++)
			{
				T sum = y[offs + 0] * _05;
				y[offs + 1] = -sum;//y[offs + 0] no need to set zero - it is ignored
				for (int j = 2; j < n - 1; j += 2)//loop from 2, not from 0(as in original version)
				{
					sum += y[offs + j];
					y[offs + j] = y[offs + j + 1];
					y[offs + j + 1] = -sum;
				}
				if (fftSizeOdd) y[offs + n - 1] = y[offs + n];//==y[offs + j] = y[offs + j + 1]; for j==n - 1
				offs += batchSize;
#if OriginalVersion
				y[offs + 0] *= 0.5;
				T sum = y[offs + 1] = 0.0;
				for (int j = 0; j < n - 1; j += 2)
				{
					sum += y[offs + j];
					y[offs + j] = y[offs + j + 1];
					y[offs + j + 1] = -sum;
				}
#endif
			}
		}

		public static void sinTransformPostprocessR2CNoSumMinus<T>(Span<T> y, int offs, int batches, int batchSize, int n) where T : IAdditionOperators<T, T, T>, IMultiplyOperators<T, double, T>
		{
			bool fftSizeOdd = (n & 1) == 1;
			//{R0, 0} {R1, I1} {R2, I2} ... {R(n+1)/2-1, I(n+1)/2-1} {Rn/2, 0} after r2c FFT, fftSize == n
			for (int k = 0; k < batches; k++)
			{
				T sum = y[offs + 0] * 0.5;
				y[offs + 1] = sum;//y[offs + 0] no need to set zero - it is ignored
				for (int j = 2; j < n - 1; j += 2)//loop from 2, not from 0(as in original version)
				{
					sum += y[offs + j];
					y[offs + j] = y[offs + j + 1];
					y[offs + j + 1] = sum;
				}
				if (fftSizeOdd) y[offs + n - 1] = y[offs + n];//==y[offs + j] = y[offs + j + 1]; for j==n - 1
				offs += batchSize;
			}
		}

		public static void sinTransformPostprocessR2HCOriginal(double[] y, int offs, int batches, int cols, int n, double coef, double[] ReImg)
		{//doesn't support odd n(fftSize)
		 //R0, R1, R2, ..., R(n+1)/2-1, Rn/2, I(n+1)/2-1, ..., I2, I1 after r2r FFT(r2HC), fftSize == n
			for (int k = 0; k < batches; k++)
			{
				//first convert to below(as from r2c)
				//{R0, 0} {R1, I1} {R2, I2} ... {R(n+1)/2-1, I(n+1)/2-1} {Rn/2, 0} after r2c FFT, fftSize == n
				for (int i = 1; i < (n + 1) / 2; i++)
				{
					ReImg[i * 2 + 0 - 2] = y[offs + i];//Re, R0 already assigned;not used Rn/2
					ReImg[i * 2 + 1 - 2] = y[offs + n - i];//Img
				}
				//Utils.printArray1DAs2D(ReImg, 1, ReImg.Length, "ReImg");
				for (int i = 2; i < n; i++) y[offs + i] = ReImg[i - 2];//R0 already assigned, skips element at index 1
				//do transform
				double sum = y[offs + 0] * 0.5;
				y[offs + 0] = 0;//y[offs + 0] no need to set zero - it is ignored
				y[offs + 1] = -sum * coef;
				for (int j = 2; j < n - 1; j += 2)
				{
					sum += y[offs + j];
					y[offs + j] = y[offs + j + 1] * coef;
					y[offs + j + 1] = -sum * coef;
				}
				offs += cols;
			}
		}

		public static void sinTransformPostprocessR2HCLessMem(double[] y, int offs, int batches, int cols, int n, double coef, double[] Img)
		{//https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
			int count = (n + 1) / 2;
			bool fftSizeOdd = (n & 1) == 1;
			//R0, R1, R2, ..., R(n+1)/2-1, Rn/2, I(n+1)/2-1, ..., I2, I1 after r2r FFT(r2HC), fftSize == n
			for (int k = 0; k < batches; k++)
			{
				//first convert to below(as from r2c)
				//{R0, 0} {R1, I1} {R2, I2} ... {R(n+1)/2-1, I(n+1)/2-1} {Rn/2, 0} after r2c FFT, fftSize == n
				for (int i = 1; i < count; i++) Img[i - 1] = y[offs + n - i];
				for (int i = count - 1; i >= 1; i--)//can only use reverse loop(else will erase data)
				{
					y[offs + i + i + 0] = y[offs + i];//not used Rn/2
					if (!fftSizeOdd || (i != count - 1)) y[offs + i + i + 1] = Img[i - 1];
				}
				//do transform
				double sum = y[offs + 0] * 0.5;
				y[offs + 0] = 0;//y[offs + 0] no need to set zero - it is ignored
				y[offs + 1] = -sum * coef;
				for (int j = 2; j < n - 1; j += 2)
				{
					sum += y[offs + j];
					y[offs + j] = y[offs + j + 1] * coef;
					y[offs + j + 1] = -sum * coef;
				}
				if (fftSizeOdd) y[offs + n - 1] = Img[count - 1 - 1] * coef;//==y[offs + j] = y[offs + j + 1] * coef; for j==n - 1
				offs += cols;
			}
		}

		public static void sinTransformPostprocessR2HCFast(double[] y, int offs, int batches, int cols, int n, double coef, double[] ReImg)
		{//doesn't support odd n(fftSize), a little bit optimized variant of sinTransformPostprocessR2HCOriginal
		 //R0, R1, R2, ..., R(n+1)/2-1, Rn/2, I(n+1)/2-1, ..., I2, I1 after r2r FFT(r2HC), fftSize == n;https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
			for (int k = 0; k < batches; k++)
			{
				//first convert to below(as from r2c)
				//{R0, 0} {R1, I1} {R2, I2} ... {R(n+1)/2-1, I(n+1)/2-1} {Rn/2, 0} after r2c FFT, fftSize == n
				for (int i = 1; i < (n + 1) / 2; i++)
				{
					ReImg[i * 2 + 0 - 2] = y[offs + i];//Re, R0 already assigned;not used Rn/2
					ReImg[i * 2 + 1 - 2] = y[offs + n - i];//Img
				}
				//do transform
				for (int i = 2; i < n - 1; i += 2) y[offs + i] = ReImg[i - 1] * coef;
				double sum = y[offs + 0] * 0.5;
				y[offs + 0] = 0;//y[offs + 0] no need to set zero - it is ignored
				y[offs + 1] = -sum * coef;
				for (int j = 2; j < n - 1; j += 2)
				{
					sum += ReImg[j - 2];
					y[offs + j + 1] = -sum * coef;
				}
				offs += cols;
			}
		}

		public static void sinTransformPostprocessR2HCLessMemVertical(double[] y, int offs, int batches, int cols, int n, double coef, double[] Img)
		{//https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
			bool fftSizeOdd = (n & 1) == 1;
			int count = (n + 1) / 2;
			//R0, R1, R2, ..., R(n+1)/2-1, Rn/2, I(n+1)/2-1, ..., I2, I1 after r2r FFT(r2HC), fftSize == n
			for (int k = 0; k < batches; k++)
			{
				//first convert to below(as from r2c)
				//{R0, 0} {R1, I1} {R2, I2} ... {R(n+1)/2-1, I(n+1)/2-1} {Rn/2, 0} after r2c FFT, fftSize == n
				for (int i = 1; i < count; i++) Img[i - 1] = y[offs + (n - i) * cols];
				for (int i = count - 1; i >= 1; i--)//can only use reverse loop(else will erase data)
				{
					y[offs + (i + i + 0) * cols] = y[offs + i * cols];//not used Rn/2
					if (!fftSizeOdd || (i != count - 1)) y[offs + (i + i + 1) * cols] = Img[i - 1];
				}
				//do transform
				double sum = y[offs + 0 * cols] * 0.5;
				y[offs + 0 * cols] = 0;//y[offs + 0] no need to set zero - it is ignored
				y[offs + 1 * cols] = -sum * coef;
				for (int j = 2; j < n - 1; j += 2)
				{
					sum += y[offs + j * cols];
					y[offs + (j + 0) * cols] = y[offs + (j + 1) * cols] * coef;
					y[offs + (j + 1) * cols] = -sum * coef;
				}
				if (fftSizeOdd) y[offs + (n - 1) * cols] = Img[count - 1 - 1] * coef;//==y[offs + j] = y[offs + j + 1] * coef; for j==n - 1
				offs += 1;//== dist
			}
		}
	}
}

using System;
using System.Numerics;

namespace VLP2D.Model
{
	public interface IFFT<T> where T : INumber<T>, ITrigonometricFunctions<T>
	{
		void SinFT(Func<int, T> input, Action<int, T> act);
		void SinFT2(Func<int, T> input, Func<int, T, T> intermed, Action<int, T> act);
		FFTCommonData<T> createCommonData();
		void asignCommonData(FFTCommonData<T> data);
	}

	public static class FFTConstant
	{
		public static int sizeOfComplex = 2;//in FFTW - typedef double fftw_complex[2];
	}

	public static class FFTCreator<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>
	{
		public static IFFT<T> createFFT(int n)
		{
			if (typeof(T) == typeof(float) || typeof(T) == typeof(double)) return new FFTWSineTransform<T>(n);
			else return new FFTLomontST<T>(n);
		}
	}

	public class FFTCommonData<T>
	{
	}

	public class FFTCalculator<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>
	{
		IFFT<T>[] fft;
		FFTCommonData<T> data;

		public FFTCalculator(int count, int points)
		{
			fft = new IFFT<T>[count];
			for (int i = 0; i < count; i++) fft[i] = FFTCreator<T>.createFFT(points);
			data = fft[0].createCommonData();
			for (int i = 0; i < count; i++) fft[i].asignCommonData(data);
		}

		public void calculate(int idx,Func<int, T> input, Action<int, T> act)
		{
			fft[idx].SinFT(input,act);
		}

		public void calculate2(int idx, Func<int, T> input, Func<int, T, T> intermed, Action<int, T> act)
		{
			fft[idx].SinFT2(input, intermed, act);
		}
	}
}

// Code to implement decently performing FFT for complex and real valued
// signals. See www.lomont.org for a derivation of the relevant algorithms 
// from first principles. Copyright Chris Lomont 2010-2012.
// This code and any ports are free for all to use for any reason as long 
// as this header is left in place.
// Version 1.1, Sept 2011

#define UseArray

using System;
using System.Numerics;
using VLP2D.Common;

/* History:
 * Sep 2011 - v1.1 - added parameters to support various sign conventions 
 *                   set via properties A and B. 
 *                 - Removed dependencies on LINQ and generic collections. 
 *                 - Added unit tests for the new properties. 
 *                 - Switched UnitTest to static.
 * Jan 2010 - v1.0 - Initial release
 */
namespace VLP2D.Model
{
	/// <summary>
	/// Represent a class that performs real or complex valued Fast Fourier 
	/// Transforms. Instantiate it and use the FFT or TableFFT methods to 
	/// compute complex to complex FFTs. Use FFTReal for real to complex 
	/// FFTs which are much faster than standard complex to complex FFTs.
	/// </summary>
	public class FFTLomontST<T> : IFFT<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>
	{
#if UseArray
		readonly T[] inoutSignal;
#endif
		int fftSizeHalf, fftSize;
		FFTCommonDataLomont<T> commonData;
		T _05 = T.CreateTruncating(0.5);

		public FFTLomontST(int nIn, bool bCreateCommonData = false)
		{
			fftSize = nIn;
			fftSizeHalf = fftSize / 2;
#if UseArray
			inoutSignal = new T[fftSize];
#endif

			if (bCreateCommonData)
			{
				FFTCommonDataLomont<T> data = (FFTCommonDataLomont<T>)createCommonData();
				asignCommonData(data);
			}
		}

		public void SinFT(Func<int, T> input, Action<int, T> act)
		{
#if !UseArray
			Span<T> inoutSignal = stackalloc T[fftSize];
#endif
			for (int i = 0; i < inoutSignal.Length; i++) inoutSignal[i] = (i > 0 && i < fftSize) ? input(i) : T.Zero;
			sinft(inoutSignal);
			for (int i = 1; i < fftSize; i++) act(i, inoutSignal[i]);
			//Utils.printArray1DAs2D(inoutSignal, 1, fftSize, "data after Lomont");
		}

		public void SinFT2(Func<int, T> input, Func<int, T, T> intermed, Action<int, T> act)
		{
#if !UseArray
			Span<T> inoutSignal = stackalloc T[fftSize];
#endif
			for (int i = 0; i < inoutSignal.Length; i++) inoutSignal[i] = (i > 0 && i < fftSize) ? input(i) : T.Zero;
			sinft(inoutSignal);
			for (int i = 0; i < inoutSignal.Length; i++) inoutSignal[i] = (i > 0 && i < fftSize) ? intermed(i, inoutSignal[i]) : T.Zero;
			sinft(inoutSignal);
			for (int i = 1; i < fftSize; i++) act(i, inoutSignal[i]);
		}

		void sinft(Span<T> data)
		{
			UtilsST.sinTransformPreprocess(data, 0, 1, 0, fftSize, commonData.wprST, commonData.wpiST);

			RealFFT(data);

			UtilsST.sinTransformPostprocessR2C(data, 0, 1, 0, fftSize);
		}

		public FFTCommonData<T> createCommonData()
		{
			commonData = new FFTCommonDataLomont<T>();
			commonData.cosTable = new T[fftSizeHalf];
			commonData.sinTable = new T[fftSizeHalf];
			InitializeTables();

			int cnt = fftSizeHalf / 4;
			if (cnt == 0) cnt = 1;
			commonData.indJ = new int[cnt];
			createJIndeses();

			(commonData.wprST, commonData.wpiST) = UtilsST.sinTransformCoeffs<T>(fftSize);
			T theta = -T.Pi / T.CreateTruncating(fftSizeHalf);
			commonData.wpr = T.Cos(theta);
			commonData.wpi = T.Sin(theta);

			return commonData;
		}

		public void asignCommonData(FFTCommonData<T> data)
		{
			commonData = (FFTCommonDataLomont<T>)data;
		}

		/// <summary>
		/// Compute the forward Fourier Transform of data, with data
		/// containing complex valued data as alternating real and imaginary 
		/// parts. The length must be a power of 2. This method caches values 
		/// and should be slightly faster on than the FFT method for repeated uses. 
		/// It is also slightly more accurate. Data is transformed in place.
		/// </summary>
		/// <param name="data">The complex data stored as alternating real 
		/// and imaginary parts</param>
		void TableFFT(Span<T> data)
		{
			if (fftSizeHalf > 1) Reverse(data); // bit index data reversal

			// do transform: so single point transforms, then doubles, etc.
			int mmax = 1;
			int tptr = 0;
			while (fftSizeHalf > mmax)
			{
				int istep = 2 * mmax;
				for (int m = 0; m < istep; m += 2)
				{
					T wr = commonData.cosTable[tptr];
					T wi = -commonData.sinTable[tptr++];
					for (int k = m; k < fftSize; k += 2 * istep)
					{
						int j = k + istep;
						T tempr = wr * data[j + 0] - wi * data[j + 1];
						T tempi = wi * data[j + 0] + wr * data[j + 1];
						data[j + 0] = data[k + 0] - tempr;
						data[j + 1] = data[k + 1] - tempi;
						data[k + 0] += tempr;
						data[k + 1] += tempi;
					}
				}
				mmax = istep;
			}
		}

		/// <summary>
		/// Compute the forwardFourier Transform of data, with 
		/// data containing real valued data only. The output is complex 
		/// valued after the first two entries, stored in alternating real 
		/// and imaginary parts. The first two returned entries are the real 
		/// parts of the first and last value from the conjugate symmetric 
		/// output, which are necessarily real. The length must be a power 
		/// of 2.
		/// </summary>
		/// <param name="data">The complex data stored as alternating real 
		/// and imaginary parts</param>
		void RealFFT(Span<T> data)
		{
			TableFFT(data);// do packed FFT. This can be changed to FFT to save memory

			T wjr = commonData.wpr;
			T wji = commonData.wpi;

			for (int j = 1; j <= fftSize / 4; ++j)
			{
				int k = fftSizeHalf - j;
				T tkr = data[2 * k + 0];    // real and imaginary parts of t_k  = t_(n/2 - j)
				T tki = data[2 * k + 1];
				T tjr = data[2 * j + 0];    // real and imaginary parts of t_j
				T tji = data[2 * j + 1];

				T a = (tjr - tkr) * wji;
				T b = (tji + tki) * wjr;
				T c = (tjr - tkr) * wjr;
				T d = (tji + tki) * wji;
				T e = (tjr + tkr);
				T f = (tji - tki);

				// compute entry y[j]
				data[2 * j] = (e + (a + b)) * _05;
				data[2 * j + 1] = (f + (d - c)) * _05;

				// compute entry y[k]
				data[2 * k] = (e - (b + a)) * _05;
				data[2 * k + 1] = ((d - c) - f) * _05;

				T temp = wjr;
				// todo - allow more accurate version here? make option?
				wjr = wjr * commonData.wpr - wji * commonData.wpi;
				wji = temp * commonData.wpi + wji * commonData.wpr;
			}

			// compute final y0 and y_{N/2}, store in data[0], data[1]
			//T temp1 = data[0];
			data[0] += data[1];
			//data[1] = temp1 - data[1];
		}

		#region Internals

		/// <summary>
		/// Call this with the size before using the TableFFT version
		/// Fills in tables for speed. Done automatically in TableFFT
		/// </summary>
		void InitializeTables()
		{
			// forward pass
			int mmax = 1, pos = 0;
			while (fftSizeHalf > mmax)
			{
				int istep = 2 * mmax;
				T theta = T.Pi / T.CreateTruncating(mmax);
				T wr = T.One, wi = T.Zero;
				T wpi = T.Sin(theta);
				// compute in a slightly slower yet more accurate manner
				T wpr = T.Sin(theta / T.CreateTruncating(2));
				wpr = -wpr * wpr * T.CreateTruncating(2); 
				for (int m = 0; m < istep; m += 2)
				{
					commonData.cosTable[pos] = wr;
					commonData.sinTable[pos++] = wi;
					T t = wr;
					wr = wr * wpr - wi * wpi + wr;
					wi = wi * wpr + t * wpi + wi;
				}
				mmax = istep;
			}
		}

		/// <summary>
		/// Swap data indices whenever index i has binary 
		/// digits reversed from index j, where data is
		/// two doubles per index.
		/// </summary>
		/// <param name="data"></param>
		void Reverse(Span<T> data)
		{
			// bit reverse the indices. This is exercise 5 in section 
			// 7.2.1.1 of Knuth's TAOCP the idea is a binary counter 
			// in k and one with bits reversed in j
			for (int k = 0; k < fftSizeHalf; k += 4)
			{
				// Knuth R2: swap - swap j+1 and k+2^(n-1), 2 entries each
				int j = commonData.indJ[k / 4];// Knuth R4: advance j
				UtilsSwap.swap(data, j + 2, k + fftSizeHalf + 0);
				UtilsSwap.swap(data, j + 3, k + fftSizeHalf + 1);
				if (j > k)
				{
					UtilsSwap.swap(data, j + 0, k + 0);
					UtilsSwap.swap(data, j + 1, k + 1);

					UtilsSwap.swap(data, j + fftSizeHalf + 2, k + fftSizeHalf + 2);
					UtilsSwap.swap(data, j + fftSizeHalf + 3, k + fftSizeHalf + 3);
				}
			} // bit reverse loop
		}

		void createJIndeses()
		{
			int j = 0;
			commonData.indJ[0] = 0;
			for (int k = 1; k < fftSizeHalf / 4; k++)
			{
				// Knuth R4: advance j
				int h = fftSizeHalf / 2;// this is Knuth's 2^(n-1)
				while (j >= h)
				{
					j -= h;
					h /= 2;
				}
				j += h;
				commonData.indJ[k] = j;
			}
		}

		class FFTCommonDataLomont<U> : FFTCommonData<U>
		{
			/// <summary>
			/// Pre-computed sine/cosine tables for speed
			/// </summary>
			public U[] cosTable;
			public U[] sinTable;
			public int[] indJ;
			public U wprST, wpiST, wpr, wpi;
		}

		#endregion
	}
}

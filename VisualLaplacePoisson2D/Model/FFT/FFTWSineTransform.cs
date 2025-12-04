using FFTWSharp;
using System;
using System.Numerics;
using System.Runtime.InteropServices;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class FFTWSineTransform<T> : IFFT<T> where T : INumber<T>, ITrigonometricFunctions<T>//, IAdditionOperators<T, double, T>, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>, IDivisionOperators<T, double, T>
	{
		readonly int n, fftInOutSize;
		readonly T[] inoutSignal;
		GCHandle hin;
		readonly IntPtr plan;
		readonly T wpr, wpi;
		Func<int, IntPtr, IntPtr, fftw_flags, IntPtr> createPlan;
		Action<IntPtr> executePlan, destroyPlan;

		public FFTWSineTransform(int fftSize)
		{//https://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html
			createPlan = typeof(T) == typeof(float) ? fftwf.dft_r2c_1d : fftw.dft_r2c_1d;
			executePlan = typeof(T) == typeof(float) ? fftwf.execute : fftw.execute;
			destroyPlan = typeof(T) == typeof(float) ? fftwf.destroy_plan : fftw.destroy_plan;

			n = fftSize;
			fftInOutSize = (n / 2 + 1) * FFTConstant.sizeOfComplex;
			inoutSignal = new T[fftInOutSize];//DFT output satisfies the “Hermitian” redundancy

			hin = GCHandle.Alloc(inoutSignal, GCHandleType.Pinned);

			plan = createPlan(n, hin.AddrOfPinnedObject(), hin.AddrOfPinnedObject(), fftw_flags.DestroyInput);

			(wpr, wpi) = UtilsST.sinTransformCoeffs<T>(n);
		}

		~FFTWSineTransform()
		{
			destroyPlan(plan);
			hin.Free();
		}

		public void SinFT(Func<int, T> input, Action<int, T> act)
		{
			for (int i = 1; i < n; i++) inoutSignal[i] = input(i);
			sinft();
			for (int i = 1; i < n; i++) act(i, inoutSignal[i]);
		}

		public void SinFT2(Func<int, T> input, Func<int, T, T> intermed, Action<int, T> act)
		{
			for (int i = 1; i < n; i++) inoutSignal[i] = input(i);
			sinft();
			for (int i = 1; i < n; i++) inoutSignal[i] = intermed(i, inoutSignal[i]);
			sinft();
			for (int i = 1; i < n; i++) act(i, inoutSignal[i]);
		}

		void sinft()
		{
			UtilsST.sinTransformPreprocess(inoutSignal, 0, 1, 0, n, wpr, wpi);
			executePlan(plan);
			UtilsST.sinTransformPostprocessR2C<T>(inoutSignal, 0, 1, 0, n);
		}

		public FFTCommonData<T> createCommonData() { return null; }
		public void asignCommonData(FFTCommonData<T> data) { }
	}

	internal sealed class fftw
	{
		const string libName = "libfftw3-3.dll";
		[DllImport(libName, EntryPoint = "fftw_plan_dft_r2c_1d", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
		public static extern IntPtr dft_r2c_1d(int n, IntPtr input, IntPtr output, fftw_flags flags);

		[DllImport(libName, EntryPoint = "fftw_destroy_plan", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
		public static extern void destroy_plan(IntPtr plan);

		[DllImport(libName, EntryPoint = "fftw_execute", ExactSpelling = true)]
		public static extern void execute(IntPtr plan);
	}

	internal sealed class fftwf
	{
		const string libName = "libfftw3f-3.dll";
		[DllImport(libName, EntryPoint = "fftwf_plan_dft_r2c_1d", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
		public static extern IntPtr dft_r2c_1d(int n, IntPtr input, IntPtr output, fftw_flags flags);

		[DllImport(libName, EntryPoint = "fftwf_destroy_plan", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
		public static extern void destroy_plan(IntPtr plan);

		[DllImport(libName, EntryPoint = "fftwf_execute", ExactSpelling = true)]
		public static extern void execute(IntPtr plan);
	}
}

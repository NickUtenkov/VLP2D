using ManagedCuda;
using ManagedCuda.CudaFFT;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using VLP2D.Common;
using static VLP2D.Common.UtilsElapsed;

namespace VLP2D.Model
{
	class VariablesSeparationSchemeNoProgonkaCU<T> : VariablesSeparationSchemeCU<T> where T : struct, INumber<T>, IMinMaxValue<T>, ITrigonometricFunctions<T>
	{
		int maxFFTN1Vectors;
		int allFFTN1WorkSize;
		IFFTCU<T> fft, fft1, fft2;
		VariablesSeparationFFTN2CU<T> fftN2;
		VariablesSeparationFFTN1CU<T> fftN1;
		Dictionary<int, CudaFFTPlanMany> plans;

		public VariablesSeparationSchemeNoProgonkaCU(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, ParallelOptions optionsParallel, Action<double> reportProgressIn, int cudaDevice) :
			base(cXSegments, cYSegments, stepX, stepY, fKsi, optionsParallel, reportProgressIn, cudaDevice)
		{
			allFFTN2WorkSize = dim1;
			allFFTN1WorkSize = dim2;

			int memDivider = 2;//for two buffers;4 as in OpenCL
			long maxFloats = (ctx.GetFreeDeviceMemorySize() / memDivider) / Marshal.SizeOf(typeof(T));

			FFTN2RealInputSize = N2;//vector lenth for which FFT is used
			int FFTN2ComplexOutputSize = (FFTN2RealInputSize / 2 + 1) * FFTConstant.sizeOfComplex;//DFT output satisfies the “Hermitian” redundancy
			maxFFTN2Vectors = (int)Math.Min(maxFloats / FFTN2ComplexOutputSize, allFFTN2WorkSize);
			long bufSizeFFTN2 = maxFFTN2Vectors * FFTN2ComplexOutputSize;

			int FFTN1RealInputSize = N1;//vector lenth for which FFT is used
			int FFTN1ComplexOutputSize = (FFTN1RealInputSize / 2 + 1) * FFTConstant.sizeOfComplex;//DFT output satisfies the “Hermitian” redundancy
			maxFFTN1Vectors = (int)Math.Min(maxFloats / FFTN1ComplexOutputSize, allFFTN1WorkSize);
			long bufSizeFFTN1 = maxFFTN1Vectors * FFTN1ComplexOutputSize;

			long bufSize = Math.Max(bufSizeFFTN2, bufSizeFFTN1);
			try
			{
				inOutData = new CudaDeviceVariable<T>(bufSize);
				dataAux = new CudaDeviceVariable<T>(bufSize);//maxFFTN1Vectors * (FFTN2RealInputSize - 1)
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
			{
				plans = new Dictionary<int, CudaFFTPlanMany>();
				fft = new FFTCU<T>(ctx, plans, N1, true);
				fft1 = new FFTN1CU<T>(ctx, fft, N1, N2, stepX2, stepY2);
				fft2 = new FFTCU<T>(ctx, plans, N2, true);
			}
			else
			{
				fft = new FFTLomontCU<T>(ctx, N1);
				fft1 = new FFTN1CU<T>(ctx, fft, N1, N2, stepX2, stepY2);
				fft2 = new FFTLomontCU<T>(ctx, N2);
			}
			fftN1 = new VariablesSeparationFFTN1CU<T>(fft1, dim1, inOutData, un, dim2, dataAux, ctx, stepX2, stepY2);
			fftN2 = new VariablesSeparationFFTN2CU<T>(fft2, dim2, inOutData, un);
		}

		~VariablesSeparationSchemeNoProgonkaCU()
		{
		}

		override public T doIteration(int iter)
		{
			float elapsed;
			initElapsedList();

			elapsed = getExecutedSeconds(stopWatchEL, () => initRigthHandSide(fKsi, stepX, stepY));//near border values are initialized in initTopBottomBorders, initLeftRightBorders in DirectOCLScheme
			listElapsedAdd("initRHS", elapsed);

			int[] stripHeights = Utils.calculateWorkSizes(maxFFTN2Vectors, allFFTN2WorkSize);
			int[] stripWidths = Utils.calculateWorkSizes(maxFFTN1Vectors, allFFTN1WorkSize);

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2.calculate(stripHeights, T.One, showProgress, 25, areIterationsCanceled));//[SNR] p.192, (24)
			listElapsedAdd("FFTN2 1", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN1.calculate(stripWidths, showProgress, 50, areIterationsCanceled));//[SNR] p.192, (25),p.192, (26)
			listElapsedAdd("FFTN1", elapsed);
			fftN1.cleanup();
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2.calculate(stripHeights, T.CreateTruncating(4.0 / (N1 * N2)), showProgress, 25, areIterationsCanceled));//[SNR] p.192, (27)
			listElapsedAdd("FFTN2 2", elapsed);

			return base.doIteration(iter);
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		override public void cleanup()
		{
			if (plans != null) foreach (var plan in plans) plan.Value.Dispose();
			plans = null;
			fft?.cleanup();
			fft1?.cleanup();
			fft2?.cleanup();
			fftN1?.cleanup();
			fftN1 = null;
			fftN2?.cleanup();
			fftN2 = null;
			base.cleanup();
		}
	}
}

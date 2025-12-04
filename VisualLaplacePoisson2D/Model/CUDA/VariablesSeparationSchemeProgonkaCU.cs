using ManagedCuda;
using ManagedCuda.CudaFFT;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using VLP2D.Common;
using static VLP2D.Common.UtilsElapsed;

namespace VLP2D.Model
{
	class VariablesSeparationSchemeProgonkaCU<T> : VariablesSeparationSchemeCU<T> where T : struct, INumber<T>, IMinMaxValue<T>, ITrigonometricFunctions<T>, IRootFunctions<T>, ILogarithmicFunctions<T>
	{
		IFFTCU<T> fft2;
		VariablesSeparationFFTN2CU<T> fftN2;
		Dictionary<int, CudaFFTPlanMany> plans;
		VariablesSeparationProgonkaCU<T> progonka;
		int maxProgonkaVectors;
		int allProgonkaWorkSize;

		public VariablesSeparationSchemeProgonkaCU(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, ParallelOptions optionsParallel, Action<double> reportProgressIn, int cudaDevice) :
			base(cXSegments, cYSegments, stepX, stepY, fKsi, optionsParallel,reportProgressIn, cudaDevice)
		{
			allFFTN2WorkSize = dim1;//number of vectors of size FFTN2DataSize(see below)
			allProgonkaWorkSize = dim2;//number of vectors

			int memDivider = 4;//4 as in OpenCL
			ManagedCuda.BasicTypes.SizeT memFreeSize = ctx.GetFreeDeviceMemorySize();
			ManagedCuda.BasicTypes.SizeT memTotalSize = ctx.GetTotalDeviceMemorySize();
			long maxFloats = (memFreeSize / memDivider) / Marshal.SizeOf(typeof(T));//GetFreeDeviceMemorySize, GetTotalDeviceMemorySize

			FFTN2RealInputSize = (dim2 + 1);//vector lenth for which FFT is used(' + 1' for FFT padding), fftSize
			int FFTN2ComplexOutputSize = (FFTN2RealInputSize / 2 + 1) * FFTConstant.sizeOfComplex;//DFT output satisfies the “Hermitian” redundancy
			maxFFTN2Vectors = Math.Min((int)maxFloats / FFTN2ComplexOutputSize, allFFTN2WorkSize);
			long bufSizeFFTN2InOut = maxFFTN2Vectors * FFTN2ComplexOutputSize;

			maxProgonkaVectors = Math.Min((int)maxFloats / dim1, allProgonkaWorkSize);
			int maxSimultaneousThreads = UtilsCU.getMaxThreads(ctx);
			int multiple = maxProgonkaVectors / maxSimultaneousThreads;
			if (multiple > 0) maxProgonkaVectors = multiple * maxSimultaneousThreads;
			long bufSizeProgonka = dim1 * maxProgonkaVectors;

			long bufSize = Math.Max(bufSizeFFTN2InOut, bufSizeProgonka);
			try
			{
				inOutData = new CudaDeviceVariable<T>(bufSize);
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
			{
				plans = new Dictionary<int, CudaFFTPlanMany>();
				fft2 = new FFTCU<T>(ctx, plans, N2, true);
			}
			else fft2 = new FFTLomontCU<T>(ctx, N2);

			fftN2 = new VariablesSeparationFFTN2CU<T>(fft2, dim2, inOutData, un);

			progonka = new VariablesSeparationProgonkaCU<T>(ctx, inOutData, un, dim1, dim2, stepX2, stepY2, maxProgonkaVectors, allProgonkaWorkSize);
		}

		~VariablesSeparationSchemeProgonkaCU()
		{
		}

		override public T doIteration(int iter)
		{
			float elapsed;
			initElapsedList();

			elapsed = getExecutedSeconds(stopWatchEL, () => initRigthHandSide(fKsi, stepX, stepY));//near border values are initialized in initTopBottomBorders, initLeftRightBorders in DirectOCLScheme
			listElapsedAdd("initRHS", elapsed);

			int[] workSizes = Utils.calculateWorkSizes(maxFFTN2Vectors, allFFTN2WorkSize);

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2.calculate(workSizes, T.One, showProgress, 33, areIterationsCanceled));//[SNR] p.195, (37), same as [SNR] p.192, (24)
			listElapsedAdd("FFTN2 1", elapsed);
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => progonka.calculate(showProgress, 34, areIterationsCanceled));
			listElapsedAdd("progonka", elapsed);
			progonka.cleanup();
			if (areIterationsCanceled()) return T.Zero;

			elapsed = getExecutedSeconds(stopWatchEL, () => fftN2.calculate(workSizes, T.CreateTruncating(2.0 / N2), showProgress, 33, areIterationsCanceled));//[SNR] p.195, (39), same as [SNR] p.192
			listElapsedAdd("FFTN2 2", elapsed);

			return base.doIteration(iter);
		}

		override public void cleanup()
		{
			if (plans != null) foreach (var plan in plans) plan.Value.Dispose();
			plans = null;
			fft2?.cleanup();
			fft2 = null;
			fftN2?.cleanup();
			fftN2 = null;
			progonka?.cleanup();
			progonka = null;
			UtilsCU.disposeBuf(ref inOutData);
			base.cleanup();
		}

		override public string getElapsedInfo() { return timesElapsed(); }
	}
}

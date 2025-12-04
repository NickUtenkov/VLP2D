
using CLMathLibraries.CLFFT;
using Cloo;
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
	class VariablesSeparationSchemeNoProgonkaOCL<T> : VariablesSeparationSchemeOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, IMinMaxValue<T>
	{
		long maxFFTN1Vectors;
		int allFFTN1WorkSize;
		IFFTOCL<T> fft, fft1, fft2;//for use inside fftN2, fftN1
		VariablesSeparationFFTN2OCL<T> fftN2;
		VariablesSeparationFFTN1OCL<T> fftN1;
		Dictionary<int, CLFFTPlan> plans;
		BufferOCL<T> dataAux = null;

		public VariablesSeparationSchemeNoProgonkaOCL(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, ParallelOptions optionsParallel, PlatformOCL platform, DeviceOCL device, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, stepX, stepY, fKsi, optionsParallel, platform, device, reportProgressIn)
		{
			allFFTN2WorkSize = dim1;
			allFFTN1WorkSize = dim2;
			long maxFloats = UtilsCL.maxMemoryAllocationSize(device) / Marshal.SizeOf(typeof(T));

			FFTN2RealInputSize = N2;//vector lenth for which FFT is used
			int FFTN2ComplexOutputSize = (FFTN2RealInputSize / 2 + 1) * FFTConstant.sizeOfComplex;
			maxFFTN2Vectors = Math.Min((int)maxFloats / FFTN2ComplexOutputSize, allFFTN2WorkSize);
			long bufSizeFFTN2 = maxFFTN2Vectors * FFTN2ComplexOutputSize;

			int FFTN1RealInputSize = N1;//vector lenth for which FFT is used
			int FFTN1ComplexOutputSize = (FFTN1RealInputSize / 2 + 1) * FFTConstant.sizeOfComplex;
			maxFFTN1Vectors = Math.Min(maxFloats / FFTN1ComplexOutputSize, allFFTN1WorkSize);
			long bufSizeFFTN1 = maxFFTN1Vectors * FFTN1ComplexOutputSize;

			long bufSize = Math.Max(bufSizeFFTN2, bufSizeFFTN1);

			try
			{
				data = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, bufSize);
				dataAux = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, bufSize);

				if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
				{
					plans = new Dictionary<int, CLFFTPlan>();
					fft = new FFTOCL<T>(commands, plans, N1, true);
					fft1 = new FFTN1OCL<T>(commands, fft, N1, N2, stepX2, stepY2, true);
					fft2 = new FFTOCL<T>(commands, plans, N2, true);
				}
				else
				{
					fft = new FFTLomontOCL<T>(commands, N1);
					fft1 = new FFTN1OCL<T>(commands, fft, N1, N2, stepX2, stepY2, true);
					fft2 = new FFTLomontOCL<T>(commands, N2);
				}
				fftN1 = new VariablesSeparationFFTN1OCL<T>(commands, fft1, dim1, dim2, data, dataAux, un, stepX2, stepY2);
				fftN2 = new VariablesSeparationFFTN2OCL<T>(commands, fft2, dim2, data, un);
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
		}

		~VariablesSeparationSchemeNoProgonkaOCL()
		{
		}

		override public T doIteration(int iter)
		{
			float elapsed;
			initElapsedList();
			elapsed = getExecutedSeconds(stopWatchEL, () => initRigthHandSide(fKsi, stepX, stepY));//near border values are initialized in initTopBottomBorders, initLeftRightBorders in DirectOCLScheme
			listElapsedAdd("initRHS", elapsed);

			int[] workSizes = Utils.calculateWorkSizes(maxFFTN2Vectors, allFFTN2WorkSize);
			try
			{
				elapsed = getExecutedSeconds(stopWatchEL, () => fftN2.calculate(workSizes, T.One, showProgress, 30, areIterationsCanceled));//[SNR] p.192, (24)
				listElapsedAdd("FFTN2 1", elapsed);
				if (areIterationsCanceled()) return T.Zero;

				elapsed = getExecutedSeconds(stopWatchEL, () => fftN1.calculate((int)maxFFTN1Vectors, allFFTN1WorkSize, showProgress, 50, areIterationsCanceled));//[SNR] p.192, (25),p.192, (26)
				listElapsedAdd("FFTN1", elapsed);
				if (areIterationsCanceled()) return T.Zero;

				elapsed = getExecutedSeconds(stopWatchEL, () => fftN2.calculate(workSizes, T.CreateTruncating(4.0 / (N1 * N2)), showProgress, 20, areIterationsCanceled));//[SNR] p.192, (27)
				listElapsedAdd("FFTN2 2", elapsed);
			}
			catch (Exception ex)
			{
				Debug.WriteLine(ex.Message);
			}

			return base.doIteration(iter);
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		override public void cleanup()
		{
			UtilsCL.disposeBuf(ref dataAux);
			if (plans != null) foreach (var plan in plans) plan.Value.Destroy();
			plans = null;

			fft.cleanup();
			fft1.cleanup();
			fft2.cleanup();

			fftN1?.cleanup();
			fftN1 = null;
			fftN2?.cleanup();
			fftN2 = null;
			base.cleanup();
		}
	}
}

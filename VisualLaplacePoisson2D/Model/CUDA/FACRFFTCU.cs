using ManagedCuda;
using ManagedCuda.CudaFFT;
using System;
using System.Collections.Generic;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FACRFFTCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		IFFTCU<T> fft;
		Dictionary<int, CudaFFTPlanMany> plans;
		int ML;
		FACRConvertFFTM2InputOutputCU<T> convertInputOutput;
		CudaDeviceVariable<T> data;

		public FACRFFTCU(CudaContext ctx, int N2, int paramL, CudaDeviceVariable<T> unCU, CudaDeviceVariable<T> data)
		{
			this.data = data;
			ML = (N2 >> paramL);
			if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
			{
				plans = new Dictionary<int, CudaFFTPlanMany>();
				fft = new FFTCU<T>(ctx, plans, ML, true);
			}
			else fft = new FFTLomontCU<T>(ctx, ML);

			convertInputOutput = new FACRConvertFFTM2InputOutputCU<T>(ctx, ML - 1, unCU, data, N2 - 1, paramL);
		}

		public void calculate(int maxFFTM2Vectors, int allFFTM2WorkSize, T coef, Action<float> showProgress, float progressPercent, Func<bool> areIterationsCanceled)
		{
			int offsetRow = 0;
			int[] workSizes = Utils.calculateWorkSizes(maxFFTM2Vectors, allFFTM2WorkSize);

			for (int i = 0; i < workSizes.Length; i++)
			{
				convertInputOutput.convertInput(offsetRow, workSizes[i]);

				fft.calculate(data, workSizes[i], coef);

				convertInputOutput.convertOutput(offsetRow, workSizes[i]);

				offsetRow += workSizes[i];

				if (areIterationsCanceled()) return;
				showProgress(progressPercent / workSizes.Length);
			}
		}

		public void cleanup()
		{
			if (plans != null) foreach (var plan in plans) plan.Value.Dispose();
			plans = null;
			data?.Dispose();
			data = null;
			convertInputOutput?.cleanup();
			convertInputOutput = null;
			fft?.cleanup();
			fft = null;
		}
	}
}
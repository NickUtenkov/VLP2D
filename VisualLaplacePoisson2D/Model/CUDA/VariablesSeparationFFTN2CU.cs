using ManagedCuda;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VariablesSeparationFFTN2CU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		IFFTCU<T> fftN2;
		T[] un;
		CudaDeviceVariable<T> data;
		readonly int N2;//fftSize

		public VariablesSeparationFFTN2CU(IFFTCU<T> fftCU, int dim2, CudaDeviceVariable<T> data, T[] un)
		{
			this.un = un;
			this.data = data;

			N2 = dim2 + 1;

			fftN2 = fftCU;
		}

		public void calculate(int[] stripHeights, T coef, Action<float> showProgress, float progressPercent, Func<bool> areIterationsCanceled)
		{
			int srcOffsetY = 0;

			for (int i = 0; i < stripHeights.Length; i++)
			{
				UtilsCU.HorizontalFFTStripCopyToDevice(un, N2, data, srcOffsetY, stripHeights[i]);//can use stripHeights[i] as dim1(for printing arrays) in case stripHeights.Length == 1

				fftN2.calculate(data, stripHeights[i], coef);

				UtilsCU.HorizontalFFTStripCopyToHost(data, N2, un, srcOffsetY, stripHeights[i]);

				srcOffsetY += stripHeights[i];

				if (areIterationsCanceled()) return;
				showProgress(progressPercent / stripHeights.Length);
			}
		}

		public void cleanup()
		{
			un = null;
			data = null;
			fftN2 = null;
		}
	}
}

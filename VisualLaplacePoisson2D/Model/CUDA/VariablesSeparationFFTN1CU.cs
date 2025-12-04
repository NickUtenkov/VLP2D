using ManagedCuda;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VariablesSeparationFFTN1CU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		IFFTCU<T> fftN1;
		T[] un;
		CudaDeviceVariable<T> data, dataAux;
		readonly int dim1, dim2;
		VariablesSeparationConvertFFTN1InputOutputCU<T> convertInputOutput;

		public VariablesSeparationFFTN1CU(IFFTCU<T> fft1, int dim1, CudaDeviceVariable<T> data, T[] un, int dim2, CudaDeviceVariable<T> dataAux, CudaContext ctx, T stepX2, T stepY2)
		{
			this.un = un;
			this.data = data;
			this.dataAux = dataAux;
			this.dim1 = dim1;
			this.dim2 = dim2;

			int fftSize = dim1 + 1;
			int N2 = dim2 + 1;

			fftN1 = fft1;

			convertInputOutput = new VariablesSeparationConvertFFTN1InputOutputCU<T>(ctx, dim1);
		}

		public void calculate(int[] stripWidths, Action<float> showProgress, float progressPercent, Func<bool> areIterationsCanceled)
		{
			int srcOffsetX = 0, offsetI = 0;

			for (int i = 0; i < stripWidths.Length; i++)
			{
				UtilsCU.VerticalStripCopyToDevice(un, dim1, dim2, dataAux, srcOffsetX, stripWidths[i]);//host vertical vectors to device vertical vectors
				convertInputOutput.transposeWithShift(dataAux, data, stripWidths[i]);//dataAux(vertical vectors) to data(horizontal vectors) with 1 padding element

				fftN1.calculate(data, stripWidths[i], T.One);

				fftN1.calculateDivideByLyambdasSum(data, stripWidths[i], offsetI);

				fftN1.calculate(data, stripWidths[i], T.One);

				convertInputOutput.transposeWithRemovingLeftRightMargins(data, dataAux, stripWidths[i]);//src(horizontal vectors,padLeft=1,padRight=2 or 1) to dst(vertical vectors)
				UtilsCU.VerticalStripCopyToHost(dataAux, dim1, dim2, un, srcOffsetX, stripWidths[i]);

				srcOffsetX += stripWidths[i];
				offsetI += stripWidths[i];

				if (areIterationsCanceled()) return;
				showProgress(progressPercent / stripWidths.Length);
			}
		}

		public void cleanup()
		{
			un = null;
			data = null;
			dataAux = null;
			fftN1 = null;
			convertInputOutput?.cleanup();
			convertInputOutput = null;
		}
	}
}

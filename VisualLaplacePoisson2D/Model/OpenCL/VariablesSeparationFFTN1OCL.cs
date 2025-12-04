using Cloo;
using System;
using System.Numerics;
using System.Runtime.InteropServices;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VariablesSeparationFFTN1OCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CommandQueueOCL commands;
		IFFTOCL<T> fftN1;
		BufferOCL<T> data, dataAux;
		VariablesSeparationConvertFFTN1InputOutputOCL<T> convertInputOutput;
		T[,] un;
		int dim1, dim2;

		public VariablesSeparationFFTN1OCL(CommandQueueOCL commands, IFFTOCL<T> fft1, int dim1, int dim2, BufferOCL<T> data, BufferOCL<T> dataAux, T[,] un, T stepX2, T stepY2)
		{
			this.commands = commands;
			this.dim1 = dim1;
			this.dim2 = dim2;
			this.un = un;
			this.data = data;
			this.dataAux = dataAux;

			fftN1 = fft1;

			convertInputOutput = new VariablesSeparationConvertFFTN1InputOutputOCL<T>(commands, dim1);
		}

		public void calculate(int maxFFTN1Vectors, int allFFTN1WorkSize, Action<float> showProgress, float progressPercent, Func<bool> areIterationsCanceled)
		{
			int[] stripWidths = Utils.calculateWorkSizes(maxFFTN1Vectors, allFFTN1WorkSize);
			int offsetJ = 0;
			SysIntX2 srcOffset = new SysIntX2(0, 0), dstOffset = new SysIntX2(0, 0), region = new SysIntX2(stripWidths[0], dim1);

			for (int i = 0; i < stripWidths.Length; i++)
			{
				region.X = (IntPtr)stripWidths[i];
				commands.WriteToBuffer(un, data, true, srcOffset, dstOffset, region, stripWidths[i] * Marshal.SizeOf(typeof(T)), dim2 * Marshal.SizeOf(typeof(T)), null);//sourceRowPitch & destinationRowPitch are wrong interchanged
				convertInputOutput.transposeWithShift(data, dataAux, stripWidths[i]);

				//UtilsCL.printOCLBuffer(rData, commands, dim2 + 1, stripWidths[i], "data OCL");//'+1' for FFT padding
				fftN1.calculate(dataAux, stripWidths[i], T.One);
				//UtilsCL.printOCLBuffer(fftN1.getOutputArray(), commands, (dim1 + 1), stripWidths[i] * 2, "after fft OCL");

				fftN1.calculateDivideByLyambdasSum(dataAux, stripWidths[i], offsetJ);
				//UtilsCL.printOCLBuffer(rData, commands, (dim1 + 1), stripWidths[i], "after Lyambda OCL");

				fftN1.calculate(dataAux, stripWidths[i], T.One);

				convertInputOutput.transposeWithRemovingLeftRightMargins(dataAux, data, stripWidths[i]);
				commands.ReadFromBuffer(data, ref un, true, dstOffset, srcOffset, region, stripWidths[i] * Marshal.SizeOf(typeof(T)), dim2 * Marshal.SizeOf(typeof(T)), null);//sourceRowPitch & destinationRowPitch are wrong interchanged

				srcOffset.X += stripWidths[i];
				offsetJ += stripWidths[i];

				if (areIterationsCanceled()) return;
				showProgress(progressPercent / stripWidths.Length);
			}
		}

		public void cleanup()
		{
			commands = null;
			un = null;
			data = null;
			dataAux = null;
			fftN1 = null;
			convertInputOutput.cleanup();
		}
	}
}

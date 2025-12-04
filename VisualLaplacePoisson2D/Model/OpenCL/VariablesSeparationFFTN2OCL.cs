using Cloo;
using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace VLP2D.Model
{
	class VariablesSeparationFFTN2OCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		IFFTOCL<T> fftN2;
		BufferOCL<T> data;
		CommandQueueOCL commands;
		T[,] un;
		readonly int N2, dim2, fftOutSize;
		readonly long srcRowPitch, dstRowPitch;
		SysIntX2 dstOffset = new SysIntX2(1, 0);

		public VariablesSeparationFFTN2OCL(CommandQueueOCL commands, IFFTOCL<T> fft2, int dim2, BufferOCL<T> data, T[,] un)
		{
			this.commands = commands;
			this.un = un;
			this.data = data;
			this.dim2 = dim2;

			N2 = dim2 + 1;
			fftN2 = fft2;

			fftOutSize = (N2 / 2 + 1) * 2;
			srcRowPitch = dim2 * Marshal.SizeOf(typeof(T));
			dstRowPitch = fftOutSize * Marshal.SizeOf(typeof(T));//sourceRowPitch & destinationRowPitch are wrong interchanged
		}

		public void calculate(int[] workSizes, T coef, Action<float> showProgress, float progressPercent, Func<bool> areIterationsCanceled)
		{
			SysIntX2 srcOffset = new SysIntX2(0, 0), region = new SysIntX2(dim2, 0);

			for (int i = 0; i < workSizes.Length; i++)
			{
				int workSize = workSizes[i];//can use as dim1(for printing arrays) in case workSizes.Length == 1
				region.Y = (IntPtr)workSize;
				commands.WriteToBuffer(un, data, true, srcOffset, dstOffset, region, dstRowPitch, srcRowPitch, null);//UtilsCL.printOCLBuffer(cData, commands, workSize + 1, dim2, "data OCL");//'+1' for FFT padding

				fftN2.calculate(data, workSizes[i], coef);//UtilsCU.printCUDABuffer(data, workSizes[i], verctorLength * 2, "fft CU");//'+1' for FFT padding

				commands.ReadFromBuffer(data, ref un, true, dstOffset, srcOffset, region, dstRowPitch, srcRowPitch, null);//sourceRowPitch & destinationRowPitch are wrong interchanged

				srcOffset.Y += workSize;

				if (areIterationsCanceled()) return;
				showProgress(progressPercent / workSizes.Length);
			}
		}

		public void cleanup()
		{
			fftN2 = null;
			commands = null;
			data = null;
			un = null;
		}
	}
}

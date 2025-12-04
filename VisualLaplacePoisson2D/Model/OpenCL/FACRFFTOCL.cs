
using CLMathLibraries.CLFFT;
using Cloo;
using System;
using System.Collections.Generic;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class FACRFFTOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		protected CommandQueueOCL commands;
		IFFTOCL<T> fft;
		Dictionary<int, CLFFTPlan> plans;
		BufferOCL<T> data, un;//used only for printing values
		int columnsInArray;//used only for printing values
		int ML;//used only for printing values
		FACRConvertFFTM2InputOutputOCL<T> convertInputOutput;

		public FACRFFTOCL(CommandQueueOCL commands, int N2, int paramL, BufferOCL<T> data, BufferOCL<T> un)
		{
			this.commands = commands;
			this.data = data;
			this.un = un;
			this.columnsInArray = N2 - 1;

			ML = (N2 >> paramL);
			if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
			{
				plans = new Dictionary<int, CLFFTPlan>();
				fft = new FFTOCL<T>(commands, plans, ML, true);
			}
			else fft = new FFTLomontOCL<T>(commands, ML);

			convertInputOutput = new FACRConvertFFTM2InputOutputOCL<T>(commands, ML - 1, (ML / 2 + 1) * 2, un, data, columnsInArray, paramL);
		}

		public void calculate(int maxFFTM2Vectors, int allFFTM2WorkSize, T coef, Action<float> showProgress, float progressPercent, Func<bool> areIterationsCanceled)
		{
			int offsetRow = 0;
			int[] workSizes = Utils.calculateWorkSizes(maxFFTM2Vectors, allFFTM2WorkSize);

			for (int i = 0; i < workSizes.Length; i++)
			{
				int workSize = workSizes[i];

				convertInputOutput.convertInput(offsetRow, workSize);//UtilsCL.printOCLBuffer(data, commands, ML, workSize, "input data for fft");

				fft.calculate(data, workSize, coef);//UtilsCL.printOCLBuffer(data, commands, ML, workSize, "data after fft");

				convertInputOutput.convertOutput(offsetRow, workSize);//UtilsCL.printOCLBuffer(unOCL, commands, workSize, columnsInArray, "unOCL");

				offsetRow += workSize;

				if (areIterationsCanceled()) return;
				showProgress(progressPercent / workSizes.Length);
			}
		}

		public void cleanup()
		{
			commands = null;
			fft?.cleanup();
			fft = null;
			if (plans != null) foreach (var plan in plans) plan.Value.Destroy();
			plans = null;
			data = null;
			un = null;
			convertInputOutput?.cleanup();
			convertInputOutput = null;
		}
	}
}

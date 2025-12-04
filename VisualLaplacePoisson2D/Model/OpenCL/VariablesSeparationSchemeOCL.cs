using Cloo;
using System;
using System.Numerics;
using System.Threading.Tasks;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VariablesSeparationSchemeOCL<T> : Direct2DNoBoundariesScheme<T>, IScheme<T> where T : struct, INumber<T>, IMinMaxValue<T>
	{
		protected T[,] fn;
		protected readonly int N1, N2;
		protected Func<T, T, T> fKsi;
		protected T stepX, stepY;
		protected Action<double> reportProgress;
		float curProgress;
		bool iterationsCanceled;

		protected CommandQueueOCL commands;
		protected BufferOCL<T> data = null;

		protected int maxFFTN2Vectors, FFTN2RealInputSize;
		protected int allFFTN2WorkSize;

		public VariablesSeparationSchemeOCL(int cXSegments, int cYSegments, T stepX, T stepY, Func<T, T, T> fKsi, ParallelOptions optionsParallel, PlatformOCL platform, DeviceOCL device, Action<double> reportProgressIn)
			: base(cXSegments - 1, cYSegments - 1, stepX, stepY, optionsParallel)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			N1 = dim1 + 1;//can not be pow of 2(because fft for cXSegments is not used)
			N2 = dim2 + 1;//is 2^x
			fn = un;
			this.fKsi = fKsi;
			this.stepX = stepX;
			this.stepY = stepY;

			reportProgress = reportProgressIn;
			curProgress = 0;

			commands = UtilsCL.createCommandQueue(platform, device, CommandQueueFlagsOCL.None);
		}

		~VariablesSeparationSchemeOCL()
		{
		}

		protected void initRigthHandSide(Func<T, T, T> fKsi, T stepX, T stepY)
		{
			if (fKsi != null) iterate((i, j) => fn[i, j] += fKsi(stepX * T.CreateTruncating(i + 1), stepY * T.CreateTruncating(j + 1)));
		}

		virtual public T doIteration(int iter)
		{
			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		protected void showProgress(float count)
		{
			curProgress += count;
			reportProgress((int)curProgress);
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
		}

		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public virtual void cleanup()
		{
			UtilsCL.disposeQC(ref commands);
			UtilsCL.disposeBuf(ref data);
			un = null;
			fn = null;
		}

		protected bool areIterationsCanceled()
		{
			return iterationsCanceled;
		}
	}
}

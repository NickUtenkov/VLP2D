using ManagedCuda;
using System;
using System.Numerics;
using System.Threading.Tasks;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VariablesSeparationSchemeCU<T> : Direct1DNoBoundariesScheme<T>, IScheme<T> where T : unmanaged, INumber<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>
	{
		protected T[] fn;
		protected readonly int N1, N2;
		protected Func<T, T, T> fKsi;
		protected Action<double> reportProgress;
		float curProgress;
		bool iterationsCanceled;

		protected CudaContext ctx;
		protected CudaDeviceVariable<T> inOutData;//(fftSize/2+1)*complexSize * count
		protected CudaDeviceVariable<T> dataAux;//(fftSize - 1) * count

		protected int maxFFTN2Vectors, FFTN2RealInputSize;
		protected int allFFTN2WorkSize;

		public VariablesSeparationSchemeCU(RectangleData<T> rectData, Func<T, T, T> fKsi, Action<double> reportProgressIn, int cudaDevice)
			: base(rectData.cXSegments - 1, rectData.cYSegments - 1, rectData.xMin, rectData.yMin, rectData.stepX, rectData.stepY)
		{
			ctx = new CudaContext(cudaDevice);
			N1 = dim1 + 1;//can not be pow of 2(because fft for cXSegments is not used)
			N2 = dim2 + 1;//is 2^x
			fn = un;
			this.fKsi = fKsi;

			reportProgress = reportProgressIn;
			curProgress = 0;
		}

		~VariablesSeparationSchemeCU()
		{
		}

		protected void initRigthHandSide(Func<T, T, T> fKsi, T stepX, T stepY)
		{
			if (fKsi != null) iterate((i, j) => fn[i * dim2 + j] += fKsi(xMin + stepX * T.CreateTruncating(i + 1), yMin + stepY * T.CreateTruncating(j + 1)));
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
			UtilsCU.disposeBuf(ref inOutData);
			UtilsCU.disposeBuf(ref dataAux);
			un = null;
			fn = null;
			ctx?.Dispose();
			ctx = null;
		}

		protected bool areIterationsCanceled()
		{
			return iterationsCanceled;
		}
	}
}

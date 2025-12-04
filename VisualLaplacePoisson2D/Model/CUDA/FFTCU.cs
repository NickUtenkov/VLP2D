using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using System.Collections.Generic;
using System.Numerics;

namespace VLP2D.Model
{
	public interface IFFTCU<T> where T : struct
	{
		void calculate(CudaDeviceVariable<T> ioData, int workSize, T coef);
		void calculateDivideByLyambdasSum(CudaDeviceVariable<T> ioData, int workSize, int offset);
		void cleanup();
	}

	internal class FFTCU<T> : IFFTCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CudaContext ctx;
		protected int fftSize;
		SineTransformCU<T> sineTransform;
		Dictionary<int, CudaFFTPlanMany> plans;

		public FFTCU(CudaContext ctx, Dictionary<int, CudaFFTPlanMany> plans, int fftSize, bool useSineTransform)
		{
			this.ctx = ctx;
			this.fftSize = fftSize;
			this.plans = plans;

			if (useSineTransform) sineTransform = new SineTransformCU<T>(ctx, fftSize);
		}

		~FFTCU()
		{
		}

		//2.6. Advanced Data Layout
		//2.14. Caller Allocated Work Area Support
		//https://stackoverflow.com/questions/25603394/1d-batched-ffts-of-real-arrays
		public void calculate(CudaDeviceVariable<T> ioData, int workSize, T coef)
		{
			CudaFFTPlanMany plan;
			if (plans.ContainsKey(workSize)) plan = plans[workSize];
			else
			{
				int[] n = { fftSize };
				int iostride = 1;//Distance between two successive input/output elements
				int odist = (fftSize / 2 + 1), idist = odist * 2;
				cufftType fftType = typeof(T) == typeof(float) ? cufftType.R2C : cufftType.D2Z;
				plan = new CudaFFTPlanMany(1, n, workSize, fftType, null, iostride, idist, null, iostride, odist);
				plans.Add(workSize, plan);
				SizeT szPlan = plan.GetSize();
			}

			sineTransform?.preProcess(ioData, workSize);

			plan.Exec(ioData.DevicePointer);

			sineTransform?.postProcess(ioData, workSize, coef);
		}

		public void calculateDivideByLyambdasSum(CudaDeviceVariable<T> ioData, int workSize, int offset)
		{
		}

		public virtual void cleanup()
		{
			ctx = null;
			sineTransform?.cleanup();
			sineTransform = null;
			plans = null;
		}
	}
}

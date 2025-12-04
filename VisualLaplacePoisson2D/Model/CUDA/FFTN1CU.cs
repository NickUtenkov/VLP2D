using DD128Numeric;
using ManagedCuda;
using QD256Numeric;
using System.Linq;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FFTN1CU<T> : IFFTCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CudaContext ctx;
		IFFTCU<T> fft1;
		int fftSize;
		CudaDeviceVariable<T> lyambda1, lyambda2;
		CudaKernel kernel;
		object[] args;

		public FFTN1CU(CudaContext ctx, IFFTCU<T> fft1, int fftSize, int N2, T stepX2, T stepY2)
		{
			this.ctx = ctx;
			this.fft1 = fft1;
			this.fftSize = fftSize;
			createLyambaArrays(N2, fftSize, stepY2, stepX2);
			kernel = createKernel(ctx);
			args = [null, 0, fftSize - 1, (fftSize/2 + 1) * 2, lyambda1.DevicePointer, lyambda2.DevicePointer, 0];
		}

		public void calculate(CudaDeviceVariable<T> ioData, int workSize, T coef)
		{
			fft1.calculate(ioData, workSize, coef);
		}

		public void calculateDivideByLyambdasSum(CudaDeviceVariable<T> ioData, int workSize, int offsetJ)
		{
			args[0] = ioData.DevicePointer;
			args[1] = workSize;
			args[6] = offsetJ;

			UtilsCU.set2DKernelDims(kernel, workSize, fftSize - 1);
			kernel.Run(args);
		}

		void createLyambaArrays(int N1, int N2, T stepX2, T stepY2)
		{
			T[] lyambda1Tmp = calcLyambda(N1, stepX2);
			lyambda1 = lyambda1Tmp;

			if ((N1 != N2) || (T.Abs(stepX2 - stepY2) > T.CreateTruncating(1E-10)))
			{
				T[] lyambda2Tmp = calcLyambda(N2, stepY2);
				lyambda2 = lyambda2Tmp;
			}
			else
			{
				lyambda2 = lyambda1;
			}
		}

		T[] calcLyambda(int n, T step2)
		{
			T[] lyamba = new T[n - 1];
			T pi2N = T.Pi / T.CreateTruncating(n) / T.CreateTruncating(2);
			for (int i = 1; i < n; i++)
			{
				T sin = T.Sin(pi2N * T.CreateTruncating(i));
				lyamba[i - 1] = sin * sin * T.CreateTruncating(4.0) / step2;
			}

			return lyamba;
		}

		CudaKernel createKernel(CudaContext ctx)
		{
			string functionName = "lyambda";
			string args = string.Format("({0} *ioData, int dim1, int dim2, int vectorLength, {0} *lyambda1, {0} *lyambda2, int offsetI)\n", Utils.getTypeName<T>());
			string srcLyambda =
@"
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < dim1 && j < dim2)
	{
		int idx = i * vectorLength + j + 1;
		ioData[idx] = ioData[idx] / (lyambda1[offsetI + i] + lyambda2[j]);
	}
}
";
			string strProgram = UtilsCU.kernelPrefix + functionName + args + srcLyambda;
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;
			return UtilsCU.createKernel(strProgram, functionName, ctx);
		}

		public void cleanup()
		{
			ctx?.UnloadKernel(kernel);
			ctx = null;
			kernel = null;
			args = null;
			fft1 = null;
			if (lyambda2 != lyambda1) UtilsCU.disposeBuf(ref lyambda2);
			UtilsCU.disposeBuf(ref lyambda1);
		}
	}
}

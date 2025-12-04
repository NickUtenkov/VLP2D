using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FFTLomontCU<T> : FFTLomontBaseCU<T>, IFFTCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CudaContext ctx;
		object[] argsReverse, argsTableFFT, argsRealFFT;
		int fftSize;//use in DEBUG(for printCUDABuffer)

		public FFTLomontCU(CudaContext ctx, int fftSize) : base(ctx, fftSize)
		{
			this.ctx = ctx;
			this.fftSize = fftSize;
			CUmodule? module;
			string moduleName = UtilsCU.moduleName("FFTLomont_", Utils.getTypeName<T>(), ctx.DeviceId);

			string functionNameReverse = "reverseFFT";
			string functionNameTableFFT = "tableFFT";
			string functionNameRealFFT = "realFFT";

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string strProgram = "static __device__ __constant__ int fftSize, fftInOutSize, fftSizeHalf;\n";
				strProgram += string.Format("static __device__ __constant__ {0} wpr, wpi;\n", Utils.getTypeName<T>());
				strProgram += createProgramReverse(functionNameReverse);
				strProgram += createProgramTableFFT(functionNameTableFFT);
				strProgram += createProgramRealFFT(functionNameRealFFT);

				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}

			if (fftSize > 2) kernelReverse = new CudaKernel(functionNameReverse, (CUmodule)module);
			kernelTableFFT = new CudaKernel(functionNameTableFFT, (CUmodule)module);
			kernelRealFFT = new CudaKernel(functionNameRealFFT, (CUmodule)module);

			kernelTableFFT.SetConstantVariable("fftSize", fftSize);
			kernelTableFFT.SetConstantVariable("fftSizeHalf", fftSize / 2);
			kernelTableFFT.SetConstantVariable("fftInOutSize", (fftSize / 2 + 1) * 2);

			T theta = -T.Pi * T.CreateTruncating(2) / T.CreateTruncating(fftSize);
			kernelRealFFT.SetConstantVariable("wpr", T.Cos(theta));
			kernelRealFFT.SetConstantVariable("wpi", T.Sin(theta));

			if (fftSize > 2) argsReverse = [null, 0, indJ.DevicePointer];

			argsTableFFT = [null, 0, sinTable.DevicePointer, cosTable.DevicePointer];

			argsRealFFT = [null, 0];
		}

		public void calculate(CudaDeviceVariable<T> data, int workSize, T coef)
		{
			sineTransform.preProcess(data, workSize);

			if (kernelReverse != null)
			{
				argsReverse[0] = data.DevicePointer;
				argsReverse[1] = workSize;
				UtilsCU.set1DKernelDims(kernelReverse, workSize);
				kernelReverse.Run(argsReverse);
			}

			argsTableFFT[0] = data.DevicePointer;
			argsTableFFT[1] = workSize;
			UtilsCU.set1DKernelDims(kernelTableFFT, workSize);
			kernelTableFFT.Run(argsTableFFT);

			argsRealFFT[0] = data.DevicePointer;
			argsRealFFT[1] = workSize;
			UtilsCU.set1DKernelDims(kernelRealFFT, workSize);
			kernelRealFFT.Run(argsRealFFT);

			sineTransform.postProcess(data, workSize, coef);
		}

		public void calculateDivideByLyambdasSum(CudaDeviceVariable<T> ioData, int workSize, int offset)
		{
		}

		public new void cleanup()
		{
			base.cleanup();
			ctx.UnloadModule(kernelTableFFT.CUModule);
			ctx = null;
		}
	}
}

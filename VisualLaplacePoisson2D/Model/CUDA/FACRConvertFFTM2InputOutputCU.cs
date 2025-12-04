using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FACRConvertFFTM2InputOutputCU<T> where T : struct
	{
		CudaKernel kernelInput, kernelOutput;
		object[] argsInput, argsOutput;
		CudaContext ctx;
		int workSizeDim2;

		public FACRConvertFFTM2InputOutputCU(CudaContext ctx, int workSizeDim2, CudaDeviceVariable<T> unCU, CudaDeviceVariable<T> fftData, int columnsInArray, int paramL)
		{
			this.ctx = ctx;
			this.workSizeDim2 = workSizeDim2;

			int fftSize = workSizeDim2 + 1;
			int sizeofComplex = 2;
			int fftOutputSize = (fftSize / 2 + 1) * sizeofComplex;//Hermitian redundancy

			createKernels(columnsInArray, paramL, fftOutputSize, fftSize, unCU.DevicePointer, fftData.DevicePointer);
		}

		public void convertInput(int offsetRow, int workSizeDim1)
		{
			argsInput[2] = offsetRow;
			argsInput[3] = workSizeDim1;
			UtilsCU.set2DKernelDims(kernelInput, workSizeDim1, workSizeDim2);
			kernelInput.Run(argsInput);
		}

		public void convertOutput(int offsetRow, int workSizeDim1)
		{
			argsOutput[2] = offsetRow;
			argsOutput[3] = workSizeDim1;
			UtilsCU.set2DKernelDims(kernelOutput, workSizeDim1, workSizeDim2);
			kernelOutput.Run(argsOutput);
		}

		public void cleanup()
		{
			ctx.UnloadModule(kernelInput.CUModule);
			ctx = null;
		}

		void createKernels(int columnsInArray, int paramL, int fftOutputSize, int fftSize, CUdeviceptr ptrUn, CUdeviceptr ptrData)
		{
			CUmodule? module;
			string moduleName = UtilsCU.moduleName("FACRConvertFFTM2InputOutput_", Utils.getTypeName<T>(), ctx.DeviceId);
			string functionNameconvertInputM2 = "convertInputM2";
			string functionNameconvertOutputM2 = "convertOutputM2";

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string defines =
@"
static __device__ __constant__ int cols, shift, padRight;
";
				string strProgram = defines;

				strProgram += createProgramConvertInput(functionNameconvertInputM2);
				strProgram += createProgramConvertOutput(functionNameconvertOutputM2);

				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strTypeDefDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strTypeDefQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}

			kernelInput = new CudaKernel(functionNameconvertInputM2, (CUmodule)module);
			kernelOutput = new CudaKernel(functionNameconvertOutputM2, (CUmodule)module);

			kernelInput.SetConstantVariable("cols", columnsInArray);
			kernelInput.SetConstantVariable("shift", paramL);
			kernelInput.SetConstantVariable("padRight", fftOutputSize - fftSize);

			argsInput = [ptrUn, ptrData, 0, 0, workSizeDim2];
			argsOutput = [ptrData, ptrUn, 0, 0, workSizeDim2, fftOutputSize];
		}

		string createProgramConvertInput(string functionName)
		{
			string args = string.Format("({0} *src, {0} *dst, int offsetRow, int upperX, int upperY)\n", Utils.getTypeName<T>());
			string srcInput =
	@"
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;

	if (i < upperX && j <= upperY)
	{
		dst[i * (upperY + 1 + padRight) + j] = src[(i + offsetRow) * cols + (j << shift) - 1];
	}
}
";
			return UtilsCU.kernelPrefix + functionName + args + srcInput;
		}

		string createProgramConvertOutput(string functionName)
		{
			string args = string.Format("({0} *src, {0} *dst, int offsetRow, int upperX, int upperY, int fftOutputSize)\n", Utils.getTypeName<T>());
			string srcOutput =
	@"
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;

	if (i < upperX && j <= upperY)
	{
		dst[(i + offsetRow) * cols + (j << shift) - 1] = src[i * fftOutputSize + j];
	}
}
";
			return UtilsCU.kernelPrefix + functionName + args + srcOutput;
		}
	}
}

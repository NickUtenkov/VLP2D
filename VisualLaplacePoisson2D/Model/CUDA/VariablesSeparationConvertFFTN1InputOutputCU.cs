using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class VariablesSeparationConvertFFTN1InputOutputCU<T> where T : struct
	{
		CudaKernel kernelInput, kernelOutput;
		object[] argsInput, argsOutput;
		CudaContext ctx;
		int dim1;//==fftSize - 1

		public VariablesSeparationConvertFFTN1InputOutputCU(CudaContext ctx, int dim1)
		{
			this.dim1 = dim1;
			this.ctx = ctx;

			int fftSize = dim1 + 1;
			int fftInOutSize = (fftSize / 2 + 1) * FFTConstant.sizeOfComplex;//Hermitian redundancy

			createKernels(fftInOutSize);
		}

		public void transposeWithShift(CudaDeviceVariable<T> src, CudaDeviceVariable<T> dst, int workSize)//vertical to horizontal vectors
		{
			UtilsCU.set2DKernelDims(kernelInput, dim1, workSize);
			argsInput[0] = src.DevicePointer;
			argsInput[1] = dst.DevicePointer;
			argsInput[3] = workSize;
			kernelInput.Run(argsInput);
		}

		public void transposeWithRemovingLeftRightMargins(CudaDeviceVariable<T> src, CudaDeviceVariable<T> dst, int workSize)
		{
			UtilsCU.set2DKernelDims(kernelOutput, workSize, dim1);
			argsOutput[0] = src.DevicePointer;
			argsOutput[1] = dst.DevicePointer;
			argsOutput[2] = workSize;
			kernelOutput.Run(argsOutput);
		}

		public void cleanup()
		{
			ctx.UnloadModule(kernelInput.CUModule);
			ctx = null;
			kernelInput = null;
			kernelOutput = null;
			argsInput = null;
			argsOutput = null;
		}

		void createKernels(int fftInOutSize)
		{
			CUmodule? module;
			string moduleName = UtilsCU.moduleName("VarSepConvertFFTN1InOut_", Utils.getTypeName<T>(), ctx.DeviceId);
			string functionNameConvertInput = "convertInputN1";
			string functionNameConvertOutput = "convertOutputN1";

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string strProgram = "static __device__ __constant__ int fftInOutSize;\n";
				strProgram += createProgramConvertInput(functionNameConvertInput);
				strProgram += createProgramConvertOutput(functionNameConvertOutput);

				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strTypeDefDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strTypeDefQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}

			kernelInput = new CudaKernel(functionNameConvertInput, (CUmodule)module);
			kernelOutput = new CudaKernel(functionNameConvertOutput, (CUmodule)module);
			kernelInput.SetConstantVariable("fftInOutSize", fftInOutSize);

			argsInput = [null, null, dim1, 0/*workSize*/];
			argsOutput = [null, null, 0/*workSize*/, dim1];
		}

		string createProgramConvertInput(string functionName)
		{//transpose array dim1 X dim2;dim1 = fftSize - 1, dim2 = stripWidth(workSize)
			string args = string.Format("({0} *src, {0} *dst, int dim1, int dim2)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSource =
@"
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dim1 && j < dim2)
	{
		dst[j * fftInOutSize + i + 1] = src[i * dim2 + j];//transpose with shift 1 element
	}
}";
			return strProgramHeader + programSource;
		}

		string createProgramConvertOutput(string functionName)
		{//transpose array dim1 X dim2;dim1 = stripWidth(workSize), dim2 = fftSize - 1(==dim1 in input kernel)
			string args = string.Format("({0} *src, {0} *dst, int dim1, int dim2)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSource =
@"
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < dim1 && j < dim2)
	{
		dst[j * dim1 + i] = src[i * fftInOutSize + j + 1];//transpose
	}
}";
			return strProgramHeader + programSource;
		}
	}
}

using Cloo;
using DD128Numeric;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class VariablesSeparationConvertFFTN1InputOutputOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CommandQueueOCL commands;
		protected KernelOCL kernelOutput, kernelInput;
		protected long[] gWorkSizeInput = { 0, 0 }, gWorkSizeOutput = { 0, 0 };

		public VariablesSeparationConvertFFTN1InputOutputOCL(CommandQueueOCL commands, int dim1)
		{
			this.commands = commands;
			gWorkSizeInput[0] = dim1;
			gWorkSizeInput[1] = 0;//==workSize

			gWorkSizeOutput[0] = 0;//==workSize
			gWorkSizeOutput[1] = dim1;

			int fftSize = dim1 + 1;
			int fftInOutSize = (fftSize / 2 + 1) * FFTConstant.sizeOfComplex;//Hermitian redundancy

			createKernelConvertInput(fftInOutSize);
			createKernelConvertOutput(fftInOutSize);
		}

		public void transposeWithShift(BufferOCL<T> src, BufferOCL<T> dst, int workSize)
		{
			gWorkSizeInput[1] = workSize;
			kernelInput.SetMemoryArgument(0, src);
			kernelInput.SetMemoryArgument(1, dst);

			commands.Execute(kernelInput, null, gWorkSizeInput, null, null);

			commands.Finish();
		}

		public void transposeWithRemovingLeftRightMargins(BufferOCL<T> src, BufferOCL<T> dst, int workSize)
		{
			gWorkSizeOutput[0] = workSize;
			kernelOutput.SetMemoryArgument(0, src);
			kernelOutput.SetMemoryArgument(1, dst);

			commands.Execute(kernelOutput, null, gWorkSizeOutput, null, null);

			commands.Finish();
		}

		public void cleanup()
		{
			commands = null;
			UtilsCL.disposeKP(ref kernelOutput);
			UtilsCL.disposeKP(ref kernelInput);
		}

		void createKernelConvertInput(int fftInOutSize)
		{
			string functionName = "convertInputN1";
			string strDefines =
@"
#define fftInOutSize	{0}

";
			string srcInput =
	@"
(global {0} *src, global {0} *dst)
{{
	int i = get_global_id(0);
	int j = get_global_id(1);

	int workSize = get_global_size(1);
	dst[j * fftInOutSize + i + 1] = src[i * workSize + j];//transpose with shift 1 element
}}
";
			string defines = string.Format(strDefines, fftInOutSize);
			string strProgram = defines + UtilsCL.kernelPrefix + functionName + string.Format(srcInput, Utils.getTypeName<T>());
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strTypeDefDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strTypeDefQD256 + strProgram;

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, commands.Context, commands.Device);
			kernelInput = program.CreateKernel(functionName);
		}

		void createKernelConvertOutput(int fftInOutSize)
		{//grid is transposed comparing input
			string functionName = "convertOutputN1";
			string strDefines =
@"
#define fftInOutSize	{0}

";
			string srcOutput =
	@"
(global {0} *src, global {0} *dst)
{{
	int i = get_global_id(0);
	int j = get_global_id(1);

	int workSize = get_global_size(0);
	dst[j * workSize + i] = src[i * fftInOutSize + j + 1];//transpose
}}
";
			string defines = string.Format(strDefines, fftInOutSize);
			string strProgram = defines + UtilsCL.kernelPrefix + functionName + string.Format(srcOutput, Utils.getTypeName<T>());
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strTypeDefDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strTypeDefQD256 + strProgram;

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, commands.Context, commands.Device);
			kernelOutput = program.CreateKernel(functionName);
		}
	}
}

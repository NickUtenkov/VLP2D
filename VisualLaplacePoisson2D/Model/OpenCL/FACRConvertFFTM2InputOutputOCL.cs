using Cloo;
using DD128Numeric;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class FACRConvertFFTM2InputOutputOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CommandQueueOCL commands;
		KernelOCL kernelInput, kernelOutput;
		long[] workSizeInput = { 0, 0 }, workSizeOutput = { 0, 0 }, workOffsetInput = { 0, 1 }, workOffsetOutput = { 0, 1 };

		public FACRConvertFFTM2InputOutputOCL(CommandQueueOCL commands, int worksSize, int fftInOutSize, BufferOCL<T> un, BufferOCL<T> data, int columnsInArray, int paramL)
		{
			this.commands = commands;

			workSizeInput[1] = worksSize;
			workSizeOutput[1] = worksSize;

			createKernelConvertInput(un, data, fftInOutSize, columnsInArray, paramL);
			createKernelConvertOutput(data, un, fftInOutSize, columnsInArray, paramL);
		}

		public void convertInput(int offsetRow, int workSize)
		{
			workSizeInput[0] = workSize;
			kernelInput.SetValueArgument(2, offsetRow);

			commands.Execute(kernelInput, workOffsetInput, workSizeInput, null, null);

			commands.Finish();
		}

		public void convertOutput(int offsetRow, int workSize)
		{
			workSizeOutput[0] = workSize;
			kernelOutput.SetValueArgument(2, offsetRow);

			commands.Execute(kernelOutput, workOffsetOutput, workSizeOutput, null, null);

			commands.Finish();
		}

		public void cleanup()
		{
			commands = null;
			UtilsCL.disposeKP(ref kernelInput);
			UtilsCL.disposeKP(ref kernelOutput);
		}

		void createKernelConvertInput(BufferOCL<T> un, BufferOCL<T> data, int fftInOutSize, int columnsInArray, int paramL)
		{
			string definesInput =
@"
#define cols	{0}
#define shift	{1}
#define fftInOutSize	{2}

";
			string functionName = "convertInputM2";
			string args = string.Format("(global {0} *src, global {0} *dst, int offsetRow)\r\n", Utils.getTypeName<T>());
			string srcInput =
	@"
{
	int i = get_global_id(0);
	int j = get_global_id(1);//offset is 1

	dst[i * fftInOutSize + j] = src[(i + offsetRow) * cols + (j << shift) - 1];
}
";
			string defines = string.Format(definesInput, columnsInArray, paramL, fftInOutSize);
			string strProgram = defines + UtilsCL.kernelPrefix + functionName + args + srcInput;
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strTypeDefDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strTypeDefQD256 + strProgram;

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, commands.Context, commands.Device);
			kernelInput = program.CreateKernel(functionName);
			kernelInput.SetMemoryArgument(0, un);
			kernelInput.SetMemoryArgument(1, data);
		}

		void createKernelConvertOutput(BufferOCL<T> data, BufferOCL<T> un, int fftInOutSize, int columnsInArray, int paramL)
		{
			string definesOutput =
@"
#define cols	{0}
#define shift	{1}
#define fftInOutSize	{2}

";
			string functionName = "convertOutputM2";
			string args = string.Format("(global {0} *src, global {0} *dst, int offsetRow)\r\n", Utils.getTypeName<T>());
			string srcOutput =
	@"
{
	int i = get_global_id(0);
	int j = get_global_id(1);//offset is 1

	dst[(i + offsetRow) * cols + (j << shift) - 1] = src[i * fftInOutSize + j];
}
";
			string defines = string.Format(definesOutput, columnsInArray, paramL, fftInOutSize);
			string strProgram = defines + UtilsCL.kernelPrefix + functionName + args + srcOutput;
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strTypeDefDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strTypeDefQD256 + strProgram;

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, commands.Context, commands.Device);
			kernelOutput = program.CreateKernel(functionName);
			kernelOutput.SetMemoryArgument(0, data);
			kernelOutput.SetMemoryArgument(1, un);
		}
	}
}

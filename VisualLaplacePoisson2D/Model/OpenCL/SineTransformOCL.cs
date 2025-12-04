using Cloo;
using DD128Numeric;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class SineTransformOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CommandQueueOCL commands;
		KernelOCL kernelPreProcess, kernelPostProcess;
		long[] gWorkPreProcess = { 0 }, gWorkPostProcess = { 0 };

		public SineTransformOCL(CommandQueueOCL commands)
		{
			this.commands = commands;
		}

		public void preProcess(BufferOCL<T> data, int workSize)
		{
			gWorkPreProcess[0] = workSize;
			kernelPreProcess.SetMemoryArgument(0, data);

			commands.Execute(kernelPreProcess, null, gWorkPreProcess, null, null);

			commands.Finish();
		}

		public void postProcess(BufferOCL<T> data, int workSize, T coef)
		{
			gWorkPostProcess[0] = workSize;
			kernelPostProcess.SetMemoryArgument(0, data);
			kernelPostProcess.SetValueArgument(1, coef);

			commands.Execute(kernelPostProcess, null, gWorkPostProcess, null, null);

			commands.Finish();
		}

		public void createKernelPreProcess(DeviceOCL device, ContextOCL context, int fftSize)
		{
			string funcName = "preProcessSineTransform";
			string strProgram = srcPreProcess(funcName, fftSize);
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, context, device);
			kernelPreProcess = program.CreateKernel(funcName);

			T theta = T.Pi / T.CreateTruncating(fftSize);
			T sinHalfTheta = T.Sin(theta * T.CreateTruncating(0.5));
			T wprInitial = -sinHalfTheta * sinHalfTheta * T.CreateTruncating(2.0);
			T wpiInitial = T.Sin(theta);
			kernelPreProcess.SetValueArgument(1, wprInitial);
			kernelPreProcess.SetValueArgument(2, wpiInitial);
		}

		public void createKernelPostProcess(DeviceOCL device, ContextOCL context, int fftSize)
		{
			string funcName = "postProcessSineTransform";
			string strProgram = srcPostProcess(funcName, fftSize);
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, context, device);
			kernelPostProcess = program.CreateKernel(funcName);
		}

		public void cleanup()
		{
			commands = null;

			UtilsCL.disposeKP(ref kernelPreProcess);
			UtilsCL.disposeKP(ref kernelPostProcess);
		}

		string srcPreProcess(string funcName, int fftSize)
		{
			string args = "(global {0} *rdata, {0} wpr, {0} wpi)";
			string strProgramHeader = UtilsCL.kernelPrefix + funcName;
			string strPreProcess =
@"
{{
	{0} y1, y2, wtemp;
	{0} wi = Zero, wr = One;
	int nv = get_global_size(0);//number of vectors
	int i = get_global_id(0);
	global {0} *rDat = rdata + {3} * i;
	rDat[0] = Zero;
	for (int j = 1; j < {1}; j++)//(fftSize >> 1) + 1
	{{
		wtemp = wr;
		wr = HP(wtemp * wpr - wi * wpi + wr);
		wi = HP(wi * wpr + wtemp * wpi + wi);
		y1 = HP(wi * (rDat[j] + rDat[({2} - j)]));//fftSize
		y2 = HP(0.5 * (rDat[j] - rDat[({2} - j)]));//fftSize
		rDat[(j + 0)] = HP(y1 + y2);
		rDat[({2} - j)] = HP(y1 - y2);//fftSize
	}}
}}";
			return strProgramHeader + string.Format(args + strPreProcess, Utils.getTypeName<T>(), (fftSize >> 1) + 1, fftSize, (fftSize / 2 + 1) * 2);
		}

		string srcPostProcess(string funcName, int fftSize)
		{
			string args = "(global {0} *cdata, {0} coef)";
			string strProgramHeader = UtilsCL.kernelPrefix + funcName;
			string strPostProcess =
@"
{{
	int nv = get_global_size(0);//number of vectors
	int i = get_global_id(0);
	global {0} *cDat = cdata + {1} * i;
	cDat[0] = HP(cDat[0] * 0.5);
	{0} sum = cDat[1] = Zero;
	for (int j = 0; j < {2}; j += 2)
	{{
		sum = HP(sum + cDat[j + 0]);
		cDat[j + 0] = HP(cDat[j + 1] * coef);
		cDat[j + 1] = HP(-sum * coef);//negation
	}}
}}";
			return strProgramHeader + string.Format(args + strPostProcess, Utils.getTypeName<T>(), (fftSize / 2 + 1) * 2, fftSize - 1 + ((fftSize & 1) == 1 ? 1 : 0));//((fftSize & 1) == 1 ? 1 : 0) for not power2
		}
	}
}

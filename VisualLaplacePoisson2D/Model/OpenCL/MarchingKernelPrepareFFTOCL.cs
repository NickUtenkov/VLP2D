using Cloo;
using DD128Numeric;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelPrepareFFTOCL<T>
	{
		public static KernelOCL createKernelPrepareFFT(DeviceOCL device, ContextOCL context, int L, int fftSize, int fftInOutSize)
		{
			string functionName = "prepareFFT";
			string args = string.Format("(global {0}* __restrict__ pq0, const global {0}* __restrict__ pq1, const global {0}* __restrict__ pq2)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string defines =
@"
#define L				{0}
#define upperM			{1}
#define cols			{2}
#define upperL	(L * 2)

";
			string programSource =
@"
{{
	int l = get_global_id(0);
	int m = get_global_id(1);

	if (l < upperL && m < upperM)
	{
		int idx = (l >= L) ? l - L : l + L;
		pq0[l * cols + m + 1] = HP(pq1[idx * cols + m] - pq2[l * cols + m]);
	}
}}";
			string strDefines = string.Format(defines, L, fftSize - 1, fftInOutSize);
			string strProgram = strDefines + strProgramHeader + programSource;
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			}

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, context, device);
			return program.CreateKernel(functionName);
		}
	}
}

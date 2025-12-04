using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class SineTransformCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		CudaKernel kernelPreProcess, kernelPostProcess;
		object[] argsPreProcess, argsPostProcess;
		CudaContext ctx;
		string moduleFileName(int fftSize) => UtilsCU.moduleName("SineTransform_", Utils.getTypeName<T>() + ((fftSize & 1) == 1 ? "_odd" : "_even"), ctx.DeviceId);

		public SineTransformCU(CudaContext ctx, int fftSize)
		{
			this.ctx = ctx;
			CUmodule? module;

			string functionNamePre = "preProcessSineTransform";
			string functionNamePost = "postProcessSineTransform";

			string moduleName = moduleFileName(fftSize);
			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string strProgram = "static __device__ __constant__ int fftSize, fftInOutSize, fftInOutSizeHalf, fftSizeM1;\n";
				strProgram += string.Format("static __device__ __constant__ {0} wpr, wpi;\n", Utils.getTypeName<T>());
				strProgram += srcPreProcess(functionNamePre);
				strProgram += srcPostProcess(functionNamePost, fftSize);

				if (typeof(T) == typeof(float)) strProgram = HighPrecisionCU.strSingleDefines + strProgram;
				if (typeof(T) == typeof(double)) strProgram = HighPrecisionCU.strDoubleDefines + strProgram;
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}

			kernelPreProcess = new CudaKernel(functionNamePre, (CUmodule)module);
			kernelPostProcess = new CudaKernel(functionNamePost, (CUmodule)module);

			int fftInOutSizeHalf = fftSize / 2 + 1;
			kernelPreProcess.SetConstantVariable("fftSize", fftSize);
			kernelPreProcess.SetConstantVariable("fftInOutSize", fftInOutSizeHalf * 2);
			kernelPreProcess.SetConstantVariable("fftInOutSizeHalf", fftInOutSizeHalf);
			kernelPostProcess.SetConstantVariable("fftSizeM1", fftSize - 1);

			T theta = T.Pi / T.CreateTruncating(fftSize);
			T sinHalfTheta = T.Sin(theta * T.CreateTruncating(0.5));
			kernelPreProcess.SetConstantVariable("wpr", sinHalfTheta * sinHalfTheta * T.CreateTruncating(-2.0));
			kernelPreProcess.SetConstantVariable("wpi", T.Sin(theta));

			argsPreProcess = new object[2];
			argsPostProcess = new object[3];
		}

		public void preProcess(CudaDeviceVariable<T> data, int workSize)
		{
			UtilsCU.set1DKernelDims(kernelPreProcess, workSize);
			argsPreProcess[0] = data.DevicePointer;
			argsPreProcess[1] = workSize;
			kernelPreProcess.Run(argsPreProcess);
		}

		public void postProcess(CudaDeviceVariable<T> data, int workSize, T coef)
		{
			UtilsCU.set1DKernelDims(kernelPostProcess, workSize);
			argsPostProcess[0] = data.DevicePointer;
			argsPostProcess[1] = workSize;
			argsPostProcess[2] = coef;
			kernelPostProcess.Run(argsPostProcess);
		}

		public void cleanup()
		{
			ctx.UnloadModule(kernelPreProcess.CUModule);
			ctx = null;
			argsPreProcess = null;
			argsPostProcess = null;
		}

		string srcPreProcess(string funcName)
		{
			string args = "({0} *rdata, int nv)";//nv - number of vectors
			string strHeader = UtilsCU.kernelPrefix + funcName;
			string strPreProcess =
@"
{{
	{0} y1, y2, wtemp;
	{0} wi = Zero, wr = One;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nv)
	{{
		{0} *rDat = rdata + fftInOutSize * i;
		rDat[0] = Zero;
		for (int j = 1; j < fftInOutSizeHalf; j++)
		{{
			wr = (wtemp = wr) * wpr - wi * wpi + wr;
			wi = wi * wpr + wtemp * wpi + wi;
			y1 = wi * (rDat[j] + rDat[(fftSize - j)]);
			y2 = 0.5 * (rDat[j] - rDat[(fftSize - j)]);
			rDat[(j + 0)] = y1 + y2;
			rDat[(fftSize - j)] = y1 - y2;
		}}
	}}
}}";
			return strHeader + string.Format(args + strPreProcess, Utils.getTypeName<T>());
		}

		string srcPostProcess(string funcName, int fftSize)
		{
			string args = "({0} *cdata, int nv, {0} coef)";
			string strHeader = UtilsCU.kernelPrefix + funcName;
			string strPostProcess =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nv)
	{{
		{0} *cDat = cdata + fftInOutSize * i;
		cDat[0] *= 0.5;
		{0} sum = cDat[1] = Zero;
		for (int j = 0; j < fftSizeM1; j += 2)
		{{
			sum += cDat[j + 0];
			cDat[j + 0] = cDat[j + 1] * coef;
			cDat[j + 1] = -sum * coef;
		}}
		{1}
	}}
}}";
			string lastAssignment = (fftSize & 1) == 1 ? string.Format("cDat[{0}] = cDat[{1}] * coef;", fftSize - 1, fftSize) : "//empty string, fftSize is even";
			return strHeader + string.Format(args + strPostProcess, Utils.getTypeName<T>(), lastAssignment);
		}
	}
}

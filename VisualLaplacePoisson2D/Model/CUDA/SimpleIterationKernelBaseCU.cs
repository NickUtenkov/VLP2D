using ManagedCuda;
using ManagedCuda.BasicTypes;
using VLP2D.Common;

namespace VLP2D.Model
{
	class SimpleIterationKernelBaseCU
	{
		CudaContext ctx;
		protected string strTypeName;
		protected string programSource;
		int sharedDimX, sharedDimY;
		readonly string constantsInt =
@"
#define sharedDimX {0}
#define sharedDimY {1}
static __device__ __constant__ int lastBlockSizeX, lastBlockSizeY;
static __device__ __constant__ int dimY, upperX, upperY;
";
		readonly string constantsExtra = @"static __device__ __constant__ {0} {1};";

		public SimpleIterationKernelBaseCU(CudaContext ctx, string strTypeName, int sharedDimX, int sharedDimY)
		{
			this.ctx = ctx;
			this.strTypeName = strTypeName;
			this.sharedDimX = sharedDimX;
			this.sharedDimY = sharedDimY;
		}

		protected CudaKernel createKrnl(string functionName, string argsIn, string strAction, string additionalConstants = null)
		{
			CUmodule? module;
			string moduleNamePrefix = string.Format("{0}_SharedMemory{1}X{2}", functionName, sharedDimX, sharedDimY);
			string moduleName = UtilsCU.moduleName(moduleNamePrefix, strTypeName, ctx.DeviceId);

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string args = string.Format(argsIn, strTypeName);
				string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;

				string strConstants = string.Format(constantsInt, sharedDimX, sharedDimY);
				if (additionalConstants != null) strConstants += string.Format(constantsExtra, strTypeName, additionalConstants);
				string strProgram = strConstants + strProgramHeader + formatSource(strAction);
				if (strTypeName == "DD128") strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
				if (strTypeName == "QD256") strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName, false);
			}

			return new CudaKernel(functionName, (CUmodule)module);
		}

		protected virtual string formatSource(string strAction)
		{
			return null;
		}
	}
}

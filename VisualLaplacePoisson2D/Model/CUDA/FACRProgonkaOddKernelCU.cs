using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class FACRProgonkaOddKernelCU<T>
	{
		public static CudaKernel createKernelMeetingProgonkaOdd(CudaContext ctx)
		{
			CUmodule? module;
			string functionName = "progonkaMeetingM2Odd";
			string moduleName = UtilsCU.moduleName("FACRProgonkaOdd_", Utils.getTypeName<T>(), ctx.DeviceId);

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string definesProgonka =
@"
#define ind(i, k) (((i) < alfaCounts[k]) ? (i) : alfaCounts[k] - 1)
#define indU(i)	(i) * dimY + (j)

static __device__ __constant__ int U, midX, dimY, paramL;
static __device__ __constant__ {0} stepX2, x2DivY2;

typedef {0}(*pFunc)({0} *, int, int, int);//un, i1PlusJ, j, idxDelta

__device__ {0} funcRHS1({0} *un, int i1PlusJ, int j, int idxDelta)
{{
	return stepX2 * un[i1PlusJ] + x2DivY2 * (((j - idxDelta > 0) ? un[i1PlusJ - idxDelta] : Zero) + ((j + idxDelta < dimY - 1) ? un[i1PlusJ + idxDelta] : Zero));
}}

__device__ {0} funcRHS2({0} *un, int i1PlusJ, int unused1, int unused2)
{{
	return x2DivY2 * un[i1PlusJ];
}}
";
				string srcProgonkaMeet =
@"
__device__ void progonka({0} *un, int j, int k, int idxDelta, {0} *alfa, int *alfaCounts, pFunc func)
{{
	//calc beta, put to un
	for (int i = 0; i <= midX - 1; i++)
	{{
		un[indU(i + 0)] = (func(un, indU(i + 0), j, idxDelta) + (i != 0 ? un[indU(i - 1)] : Zero)) * alfa[ind(i, k)];
		un[indU(U - i)] = (func(un, indU(U - i), j, idxDelta) + (i != 0 ? un[indU(U - (i - 1))] : Zero)) * alfa[ind(i, k)];
	}}

	un[indU(midX)] = (un[indU(midX)] + alfa[ind(midX - 1, k)] * un[indU(midX - 1)]) / (1.0 - alfa[ind(midX - 1, k)] * alfa[ind(midX - 1, k)]);//[SNR] p.77(bottom)

	//from middle to left, then from middle to right
	for (int i = midX - 1,i1 = i * dimY; i >= 0; i--,i1 -= dimY) un[i1 + j] += alfa[ind(i + 0, k)] * un[i1 + dimY + j];
	for (int i = midX + 1,i1 = i * dimY; i <= U; i++,i1 += dimY) un[i1 + j] += alfa[ind(U - i, k)] * un[i1 - dimY + j];
}}";
				string srcProgonka =
@"({0} *un, {0} *alfa, int *alfaOffsets, int *alfaCounts, int l, int idxDelta, int workSize)
{{
	int j0 = blockDim.x * blockIdx.x + threadIdx.x;
	if (j0 >= workSize) return;

	int j = ((j0 + 1) << l) - idxDelta - 1;//odd in CPU variant(because of boundaries)

	progonka(un, j, 0, idxDelta, alfa, alfaCounts, funcRHS1);
	if (paramL > 1)
	{{
		for (int k = 1; k < 1 << (l - 1); k++)
		{{
			progonka(un, j, k, idxDelta, alfa + alfaOffsets[k], alfaCounts, funcRHS2);
		}}
	}}
}}
";
				string strProgram = string.Format(definesProgonka + srcProgonkaMeet + UtilsCU.kernelPrefix + functionName + srcProgonka, Utils.getTypeName<T>());
				if (typeof(T) == typeof(float)) strProgram = HighPrecisionCU.strSingleDefines + strProgram;
				if (typeof(T) == typeof(double)) strProgram = HighPrecisionCU.strDoubleDefines + strProgram;
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}
			return new CudaKernel(functionName, (CUmodule)module);
		}
	}
}

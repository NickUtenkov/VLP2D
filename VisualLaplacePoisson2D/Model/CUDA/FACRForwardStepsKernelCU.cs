using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class FACRForwardStepsKernelCU<T> where T : struct
	{
		public static CudaKernel createKernelForwardSteps(int dim1, int dim2, CudaContext ctx)
		{
			CUmodule? module;
			string functionName = "forwardLSteps";
			string moduleName = UtilsCU.moduleName("FACRForwardSteps_", Utils.getTypeName<T>(), ctx.DeviceId);

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string definesKernel =
@"
#define fn(i,j) fn[(i) * dim2 + (j)]
#define sum(i,j,n) (fn((i), (j) - (n)) + fn((i), (j) + (n)))

static __device__ __constant__ int ub1, dim1, dim2;
static __device__ __constant__ {0} hYX2;
";
				string srcMult =
	@"extern ""C"" __device__ {0} *matricesCMultipleVector(int colFn, {0} *fn, {0} *src, {0} *dst, int cMatrices, {0} *diag)
{{
	for (int j = 0; j <= ub1; j++) src[j] = fn(j, colFn);//ub1 - upper bound on 1st dimension

	for (int k = 0; k < cMatrices; k++)
	{{
		dst[0] = diag[k] * src[0] - hYX2 * (/*0.0 + */src[1]);//0.0 == src[-1], which doesn't exist
		for (int j = 1; j < ub1; j++)
		{{
			dst[j] = diag[k] * src[j] - hYX2 * (src[j - 1] + src[j + 1]);
		}}
		dst[ub1] = diag[k] * src[ub1] - hYX2 * (src[ub1 - 1]/* + 0.0*/);//0.0 == src[ub1 + 1], which doesn't exist

		{0} *tmp = src;
		src = dst;
		dst = tmp;
	}}
	return src;
}}
";
				string srcKernel =
	@"({0} *fn, {0} *multiplied, {0} *accum, {0} *diag, int l, int n, int workSize)
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
	if (idx <= workSize)
	{{
		int j = (idx << l) - 1;
		int row = idx - 1;

		int offs = row * dim1;
		{0} *res = matricesCMultipleVector(j, fn, multiplied + offs, accum + offs, n, diag);
		{0} prev = sum(0, j, n) + res[0];
		for (int i = 1; i < ub1; i++)
		{{
			{0} cur = sum(i, j, n) + res[i];//[SNR] p.202, (19) and p.201, (15)
			fn(i - 1, j) = prev;
			prev = cur;
		}}
		fn(ub1 - 1, j) = prev;
		fn(ub1, j) = sum(ub1, j, n) + res[ub1];
	}}
}}";
				string strProgram = string.Format(definesKernel + srcMult + UtilsCU.kernelPrefix + functionName + srcKernel, Utils.getTypeName<T>());
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}
			return new CudaKernel(functionName, (CUmodule)module);
		}
	}
}

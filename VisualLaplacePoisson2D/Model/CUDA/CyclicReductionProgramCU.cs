using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class CyclicReductionProgramCU
	{
		public static string constants =
@"
static __device__ __constant__ {0} *alfaIn, *diagElements, *alphaCoefs, *un, *resIn, *sumIn, *acIn;
static __device__ __constant__ int *alfaOffsets, *alfaCounts;
static __device__ __constant__ {0} bCoef;
static __device__ __constant__ int alfaSize, dimY, dimAlfa, dimReverse;
#define ind(i) (((i) < alfaCounts[m - 2 + l]) ? (i) : alfaCounts[m - 2 + l] - 1)
";
		static string strProgonka =
@"{0} alp = alphaCoefs[m - 2 + l];//used inside format placeholders
		{0} *alfa = alfaIn + alfaOffsets[m - 2 + l];
		for (int i = 0; i < alfaSize; i++) res[i] = alfa[ind(i)] * (bCoef * {1} + ((i > 0) ? res[i - 1] : Zero));
		for (int i = alfaSize - 2; i >= 0; i--) res[i] += alfa[ind(i)] * res[i + 1];
";
		static string strSequentialSum =
@"int rowRes = (idx * m) * alfaSize;
		for (int l = 0; l < m; l++) 
		{
			decltype(resIn) res = resIn + rowRes;
			for (int i = 0; i < alfaSize; i++) unj0[i] += res[i];//[SNR] p.138, (23)(also see below)
			rowRes += alfaSize;
		}
";
		static string strCascadeSum =
@"
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + tid;

	int rowRes = (idx * m + gid) * alfaSize;
	decltype(resIn) res = resIn + rowRes;
	decltype(sumIn) sum = sumIn + (idx * m + gid + 0) * alfaSize;
	for (int i = 0; i < alfaSize; i++) sum[i] = res[i];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			decltype(sumIn) src = sumIn + (idx * m + gid + s) * alfaSize;
			for (int i = 0; i < alfaSize; i++) sum[i] += src[i];
		}

		__syncthreads();
	}
";
		public static string createAlfaProgram(string functionName, string typeName)
		{
			string args = "()";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceStoredAlfa =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < dimAlfa)
	{{
		{0} diagElem = diagElements[idx];
		{0} *alfa = alfaIn + alfaOffsets[idx];
		alfa[0] = 1.0f / diagElem;//[SNR] p.145 bottom
		for (int i = 1; i < alfaCounts[idx]; i++) alfa[i] = 1.0f / (diagElem - alfa[i - 1]);//[SNR] p.145 bottom
	}}
}}";
			return strProgramHeader + string.Format(programSourceStoredAlfa, typeName);
		}

		public static string createDirectProgram(string functionName, string typeName)
		{
			string args = "(int t, int m, int dim)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceDirect =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < dim)
	{{
		int nGroup = idx / m;
		int j = t * (nGroup + 1);
		int l = idx % m + 1;
		{1} *unjm = un + dimY * (j - m) + 1;//j minus m
		{1} *unjp = un + dimY * (j + m) + 1;//j plus m
		{1} *unj0 = un + dimY * j + 1;//j plus 0

		int rowRes = idx * alfaSize;
		{1} *res = resIn + rowRes;
		//progonka
		{0}
		if (m == 1) for (int i = 0; i < alfaSize; i++) unj0[i] = (unj0[i] + res[i]) * 0.5f;
	}}
}}";
			string progonka = string.Format(strProgonka, typeName, "alp * (unjm[i] + unjp[i])");
			return strProgramHeader + string.Format(programSourceDirect, progonka, typeName);
		}

		public static string createDirectSequentialSumProgram(string functionName, string typeName)
		{
			string args = "(int t, int m, int dim)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceDirectSequentialSum =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < dim)
	{{
		int j = t * (idx + 1);
		{0} *unj0 = un + dimY * j + 1;//j plus 0
		{1}
		for (int i = 0; i < alfaSize; i++) unj0[i] *= 0.5f;//[SNR] p.138, (23)
	}}
}}";
			return strProgramHeader + string.Format(programSourceDirectSequentialSum, typeName, strSequentialSum);
		}

		public static string createDirectCascadeSumProgram(string functionName, string typeName)
		{
			string args = "(int idx, int t, int m, int maxBlocks)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceDirectCascadeSum =
@"
{{
	if (blockDim.x * blockIdx.x + threadIdx.x < m)
	{{
	{1}
		if (tid == 0 && maxBlocks <= 1)
		{{
			int j = t * (idx + 1);
			{0} *unj0 = un + dimY * j + 1;//j plus 0
			for (int i = 0; i < alfaSize; i++) unj0[i] = (unj0[i] + sum[i]) * 0.5f;//[SNR] p.138, (23)
		}}
	}}
}}";
			return strProgramHeader + string.Format(programSourceDirectCascadeSum, typeName, strCascadeSum);
		}

		public static string createDirectCascadeSumBlocksProgram(string functionName, string typeName)
		{
			string args = "(int idx, int t, int m, int nBlocks, int threadsPerBlock)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceDirectCascadeSumBlocks =
@"
{{
	int j = t * (idx + 1);
	{0} *unj0 = un + dimY * j + 1;//j plus 0
	for (int blockIdx = 0; blockIdx < nBlocks; blockIdx++)
	{{
		{0} *sum = sumIn + (idx * m + threadsPerBlock * blockIdx) * alfaSize;
		for (int i = 0; i < alfaSize; i++) unj0[i] += sum[i];//[SNR] p.138, (23)
	}}
	for (int i = 0; i < alfaSize; i++) unj0[i] *= 0.5f;
}}";
			return strProgramHeader + string.Format(programSourceDirectCascadeSumBlocks, typeName);
		}

		public static string createPreReverseProgram(string functionName, string typeName)
		{
			string args = "(int t, int m, int dim)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourcePreReverse =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < dim)
	{{
		int j = m + t * idx;
		{0} *unj0 = un + dimY * j + 1;
		{0} *ac = acIn + alfaSize * idx;
		for (int i = 0; i < alfaSize; i++)
		{{
			ac[i] = unj0[i];
			unj0[i] = Zero;
		}}
	}}
}}";
			return strProgramHeader + string.Format(programSourcePreReverse, typeName);
		}

		public static string createReverseProgram(string functionName, string typeName)
		{
			string args = "(int t, int m)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceReverse =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < dimReverse)
	{{
		int nGroup = idx / m;
		int j = m + nGroup * t;
		int l = idx % m + 1;
		{0} *unjm = un + dimY * (j - m) + 1;//j minus m
		{0} *unjp = un + dimY * (j + m) + 1;//j plus m
		{0} *unj0 = un + dimY * j + 1;//j plus 0

		int rowRes = idx * alfaSize;
		{0} *res = resIn + rowRes;
		{0} *ac = acIn + nGroup * alfaSize;
		//progonka
		{1}
		if (m == 1) for (int i = 0; i < alfaSize; i++) unj0[i] = res[i];
	}}
}}";
			string progonka = string.Format(strProgonka, typeName, "(ac[i] + alp * (unjm[i] + unjp[i]))");
			return strProgramHeader + string.Format(programSourceReverse, typeName, progonka);
		}

		public static string createReverseSequentialSumProgram(string functionName, string typeName)
		{
			string args = "(int t, int m, int dim)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceReverseSequentialSum =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx <  dim)
	{{
		int j = m + idx * t;
		{0} *unj0 = un + dimY * j + 1;//j plus 0
		{1}
	}}
}}";
			return strProgramHeader + string.Format(programSourceReverseSequentialSum, typeName, strSequentialSum);
		}

		public static string createReverseCascadeSumProgram(string functionName, string typeName)
		{
			string args = "(int idx, int t, int m, int maxBlocks)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceReverseCascadeSum =
@"
{{
	if (blockDim.x * blockIdx.x + threadIdx.x < m)
	{{
	{1}
		if (tid == 0 && maxBlocks <= 1)
		{{
			int j = m + idx * t;
			{0} *unj0 = un + dimY * j + 1;//j plus 0
			for (int i = 0; i < alfaSize; i++) unj0[i] = sum[i];
		}}
	}}
}}";
			return strProgramHeader + string.Format(programSourceReverseCascadeSum, typeName, strCascadeSum);
		}

		public static string createReverseCascadeSumBlocksProgram(string functionName, string typeName)
		{
			string args = "(int idx, int t, int m, int nBlocks, int threadsPerBlock)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceReverseCascadeSumBlocks =
@"
{{
	int j = m + idx * t;
	{0} *unj0 = un + dimY * j + 1;//j plus 0
	for (int blockIdx = 0; blockIdx < nBlocks; blockIdx++)
	{{
		{0} *sum = sumIn + (idx * m + threadsPerBlock * blockIdx) * alfaSize;
		for (int i = 0; i < alfaSize; i++) unj0[i] += sum[i];
	}}
}}";
			return strProgramHeader + string.Format(programSourceReverseCascadeSumBlocks, typeName);
		}
	}
}

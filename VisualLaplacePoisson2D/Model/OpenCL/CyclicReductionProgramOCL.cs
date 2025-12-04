using Cloo;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class CyclicReductionProgramOCL
	{
		int N2;
		ContextOCL context;
		DeviceOCL device;
		string strTypeName;
		string strDefinesProgonka, strDefinesSum, strCascadeSum;
		string definesProgonka =
@"
#define Ny	{0}
#define alfaYUpperBound	(Ny - 2)
#define alfaYUpperBound1	(alfaYUpperBound + 1)
#define progonkaYUpperBound	alfaYUpperBound
#define dimY	(Ny + 1)
#define ind(i) (((i) < alfaCounts[m - 2 + l]) ? (i) : alfaCounts[m - 2 + l] - 1)
#define identity(a)	(a)
";
		string strProgonka =
@"{0} alp = alphaCoefs[m - 2 + l];//used inside format placeholders
	global {0} *alfa = alfaIn + alfaOffsets[m - 2 + l];
	for (int i = 0; i <= progonkaYUpperBound; i++) res[i] = HP(alfa[ind(i)] * (bCoef * {1} + identity((i > 0) ? res[i - 1] : Zero)));
	for (int i = progonkaYUpperBound - 1; i >= 0; i--) res[i] = HP(res[i] + alfa[ind(i)] * res[i + 1]);
";
		string definesSum =
@"
#define Ny	{0}
#define alfaYUpperBound1	(Ny - 1)
#define dimY	(Ny + 1)

";
		string strSequentialSum =
@"
	int rowRes = (idx * m) * alfaYUpperBound1;
	for (int l = 0; l < m; l++) 
	{{
		for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = HP(unj0[i] + resIn[rowRes + i]);//[SNR] p.138, (23)(also see below)
		rowRes += alfaYUpperBound1;
	}}
";
		string strCascadeSum0 = 
@"
	int tid = get_local_id(0);
	int gid = get_local_size(0) * get_group_id(0) + tid;

	int rowRes = (idx * m + gid) * alfaYUpperBound1;
	global {0} *res = resIn + rowRes;
	global {0} *sum = sumIn + (idx * m + gid) * alfaYUpperBound1;
	for (int i = 0; i < alfaYUpperBound1; i++) sum[i] = res[i];
	barrier(CLK_GLOBAL_MEM_FENCE);

	for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1)
	{{
		if (tid < s)
		{{
			global {0} *src = sumIn + (idx * m + gid + s) * alfaYUpperBound1;
			for (int i = 0; i < alfaYUpperBound1; i++) sum[i] = HP(sum[i] + src[i]);
		}}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}}
";
		public CyclicReductionProgramOCL(int N2, string strTypeName, ContextOCL context, DeviceOCL device)
		{
			this.N2 = N2;
			this.strTypeName = strTypeName;
			this.context = context;
			this.device = device;

			strDefinesProgonka = string.Format(definesProgonka, N2);
			strDefinesSum = string.Format(definesSum, N2);
			strCascadeSum = string.Format(strCascadeSum0, strTypeName);
		}

		public KernelOCL createAlfaProgram()
		{
			string functionName = "calcStoredAlfa";
			string args = "(global {0} *alfaIn, global int *alfaOffsets, global int *alfaCounts, global {0} *diagElements)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string definesStoredAlfa =
@"
#define alfaYUpperBound		{0}
#define alfaYUpperBound1	(alfaYUpperBound + 1)

";
			string srcStoredAlfa =
@"
{{
	int idx = get_global_id(0);
	{0} diagElem = diagElements[idx];
	global {0} *alfa = alfaIn + alfaOffsets[idx];

	alfa[0] = HP(One / diagElem);//[SNR] p.145 bottom
	for (int i = 1; i < alfaCounts[idx]; i++) alfa[i] = HP(One / (diagElem - alfa[i - 1]));//[SNR] p.145 bottom
}}";
			string strProgram = string.Format(definesStoredAlfa, N2 - 2) + strProgramHeader + string.Format(args + srcStoredAlfa, strTypeName);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createDirectProgram()
		{
			string functionName = "Direct";
			string args = "(int t, int m, global {0} *alfaIn, global int *alfaOffsets, global int *alfaCounts, global {0} *alphaCoefs, {0} bCoef, global {0} *un, global {0} *resIn)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string programSourceDirect =
@"
{{
	int idx = get_global_id(0);
	int nGroup = idx / m;
	int j = t * (nGroup + 1);
	int l = idx % m + 1;
	global {0} *unjm = un + dimY * (j - m) + 1;//j minus m
	global {0} *unjp = un + dimY * (j + m) + 1;//j plus m
	global {0} *unj0 = un + dimY * j + 1;//j plus 0

	int rowRes = idx * alfaYUpperBound1;
	global {0} *res = resIn + rowRes;
	int rowAlfa = (m - 2 + l) * alfaYUpperBound1;//used inside progonka
	//progonka
	{1}
	if (m == 1) for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = HP((unj0[i] + res[i]) * 0.5);
}}";

			string progonka = string.Format(strProgonka, strTypeName, "alp * (unjm[i] + unjp[i])");
			string strProgram = strDefinesProgonka + strProgramHeader + string.Format(args + programSourceDirect, strTypeName, progonka);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createDirectSequentialSumProgram()
		{
			string functionName = "DirectSequentialSum";
			string args = "(int t, int m, global {0} *un, global {0} *resIn)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string programSourceDirectSequentialSum =
@"
{{
	int idx = get_global_id(0);
	int j = t * (idx + 1);
	global {0} *unj0 = un + dimY * j + 1;//j plus 0

{1}
	for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = HP(unj0[i] * 0.5);//[SNR] p.138, (23)
}}";

			string strProgram = strDefinesSum + strProgramHeader + string.Format(args + programSourceDirectSequentialSum, strTypeName, strSequentialSum);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createDirectCascadeSumProgram()
		{
			string functionName = "DirectCascadeSum";
			string args = "(int idx, int t, int m, global {0} *un, global {0} *resIn, global {0} *sumIn, int maxBlocks)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string programSourceDirectCascadeSum =
@"
{{
	if (get_local_size(0) * get_group_id(0) + get_local_id(0) < m)
	{{
	{1}
		if (tid == 0 && maxBlocks <= 1)
		{{
			int j = t * (idx + 1);
			global {0} *unj0 = un + dimY * j + 1;//j plus 0
			for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = HP((unj0[i] + sum[i]) * 0.5);//[SNR] p.138, (23)
		}}
	}}
}}";
			string strProgram = strDefinesSum + strProgramHeader + string.Format(args + programSourceDirectCascadeSum, strTypeName, strCascadeSum);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createDirectCascadeSumBlocksProgram()
		{
			string functionName = "DirectCascadeSumBlocks";
			string args = "(int idx, int t, int m, global {0} *un, global {0} *sumIn, int nBlocks, int threadsPerBlock)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string programSourceDirectCascadeSumBlocks =
@"
{{
	int j = t * (idx + 1);
	global {0} *unj0 = un + dimY * j + 1;//j plus 0
	for (int blockIdx = 0; blockIdx < nBlocks; blockIdx++)
	{{
		global {0} *sum = sumIn + (idx * m + threadsPerBlock * blockIdx) * alfaYUpperBound1;
		for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = HP(unj0[i] + sum[i]);//[SNR] p.138, (23)
	}}
	for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = HP(unj0[i] * 0.5);
}}";

			string strProgram = strDefinesSum + strProgramHeader + string.Format(args + programSourceDirectCascadeSumBlocks, strTypeName);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createPreReverseProgram()
		{
			string functionName = "PreReverse";
			string args = "(int t, int m, global {0} *unIn, global {0} *acIn)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string definesPreReverse =
@"
#define Ny	{0}
#define alfaYUpperBound1	(Ny - 1)
#define dimY	(Ny + 1)

";
			string programSourcePreReverse =
@"
{{
	int idx = get_global_id(0);
	int j = m + t * idx;
	global {0} *un = unIn + dimY * j + 1;
	global {0} *ac = acIn + alfaYUpperBound1 * idx;
	for (int i = 0; i < Ny - 1; i++)
	{{
		ac[i] = un[i];
		un[i] = Zero;
	}}
}}";
			string strProgram = string.Format(definesPreReverse, N2) + strProgramHeader + string.Format(args + programSourcePreReverse, strTypeName);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createReverseProgram()
		{
			string functionName = "Reverse";
			string args = "(int t, int m, global {0} *alfaIn, global int *alfaOffsets, global int *alfaCounts, global {0} *alphaCoefs, {0} bCoef, int idxAlfaShift, global {0} *un, global {0} *resIn, global {0} *acIn)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string programSourceReverse =
@"
{{
	int idx = get_global_id(0);
	int nGroup = idx / m;
	int j = m + nGroup * t;
	int l = idx % m + 1;
	global {0} *unjm = un + dimY * (j - m) + 1;//j minus m
	global {0} *unjp = un + dimY * (j + m) + 1;//j plus m
	global {0} *unj0 = un + dimY * j + 1;//j plus 0

	int rowRes = idx * alfaYUpperBound1;
	global {0} *res = resIn + rowRes;
	global {0} *ac = acIn + idx / m * alfaYUpperBound1;
	int idxAlfa = m - 2 + l - idxAlfaShift;
	int rowAlfa = idxAlfa * alfaYUpperBound1;
	//progonka
	{1}
	if (m == 1) for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = res[i];
}}";
			string progonka = string.Format(strProgonka, strTypeName, "(ac[i] + alp * (unjm[i] + unjp[i]))");
			string strProgram = strDefinesProgonka + strProgramHeader + string.Format(args + programSourceReverse, strTypeName, progonka);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createReverseSequentialSumProgram()
		{
			string functionName = "ReverseSequentialSum";
			string args = "(int t, int m, global {0} *un, global {0} *resIn)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string programSourceReverseSequentialSum =
@"
{{
	int idx = get_global_id(0);
	int j = m + idx * t;
	global {0} *unj0 = un + dimY * j + 1;//j plus 0

{1}
}}";
			string strProgram = strDefinesSum + strProgramHeader + string.Format(args + programSourceReverseSequentialSum, strTypeName, strSequentialSum);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createReverseCascadeSumProgram()
		{
			string functionName = "ReverseCascadeSum";
			string args = "(int idx, int t, int m, global {0} *un, global {0} *resIn, global {0} *sumIn, int maxBlocks)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string programSourceReverseCascadeSum =
@"
{{
	if (get_local_size(0) * get_group_id(0) + get_local_id(0) < m)
	{{
	{1}
		if (tid == 0 && maxBlocks <= 1)
		{{
			int j = m + idx * t;
			global {0} *unj0 = un + dimY * j + 1;//j plus 0
			for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = sum[i];
		}}
	}}
}}";

			string strProgram = strDefinesSum + strProgramHeader + string.Format(args + programSourceReverseCascadeSum, strTypeName, strCascadeSum);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		public KernelOCL createReverseCascadeSumBlocksProgram()
		{
			string functionName = "ReverseCascadeSumBlocks";
			string args = "(int idx, int t, int m, global {0} *un, global {0} *sumIn, int nBlocks, int threadsPerBlock)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName;
			string programSourceReverseCascadeSumBlocks =
@"
{{
	int j = m + idx * t;
	global {0} *unj0 = un + dimY * j + 1;//j plus 0
	for (int blockIdx = 0; blockIdx < nBlocks; blockIdx++)
	{{
		global {0} *sum = sumIn + (idx * m + threadsPerBlock * blockIdx) * alfaYUpperBound1;
		for (int i = 0; i < alfaYUpperBound1; i++) unj0[i] = HP(unj0[i] + sum[i]);
	}}
}}";

			string strProgram = strDefinesSum + strProgramHeader + string.Format(args + programSourceReverseCascadeSumBlocks, strTypeName);

			ProgramOCL program = createProgram(strProgram);
			return program.CreateKernel(functionName);
		}

		ProgramOCL createProgram(string strProgram)
		{
			if (strTypeName == "float") strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (strTypeName == "double") strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
			if (strTypeName == "DD128") strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
			if (strTypeName == "QD256") strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;

			return UtilsCL.createProgram(strProgram, null, context, device);
		}
	}
}

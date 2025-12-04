using Cloo;
using DD128Numeric;
using QD256Numeric;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelReducedSystemOCL<T>
	{
		public static KernelOCL createKernelReducedSystem(DeviceOCL device, ContextOCL context, int L, int k, int M, int pqDim2)
		{
			string functionName = "solveReducedSystem";
			string args = string.Format("(global {0}* __restrict__ fi, global {0}* __restrict__ alfa, global {0}* __restrict__ pq0, {0} piLyambda, {0} subsupra2)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string defines =
@"
#define L				{0}
#define k				{1}
#define M				{2}
#define colsPQ			{3}
#define alfaUB			(L - 1)
#define overDiag(i, overUnder)	(((i & 1) == 0) ? overUnder : One)
#define underDiag(i, overUnder)	(((i & 1) == 1) ? overUnder : One)

";
			string functions = string.Format(
@"
{0} lyambdaDiv2(int n, {0} piLyambda, {0} subsupra2)
{{
	{0} sinus = sin({1});
	return HP(One + subsupra2 * sinus * sinus);//SNE p.236 under (39)
}}

void Un(int n, {0} x, {0} *out1, {0} *out2)
{{
	if (n == 0)
	{{
		*out1 = One;
		*out2 = HP(2.0 * x);
		return;
	}}
	if (n == 1)
	{{
		*out1 = HP(2.0 * x);
		*out2 = HP(4.0 * x * x - One);
		return;
	}}
	{0} u0 = Zero, u1 = One, u2 = HP(2.0 * x);
	for (int i = 0; i < n - 1; i++)
	{{
		u0 = u1;
		u1 = u2;
		u2 = HP(2.0 * x * u1 - u0);
	}}
	*out1 = u2;
	u0 = u1;
	u1 = u2;
	*out2 = HP(2.0 * x * u1 - u0);
}}

void calculateFi(int m, global {0}* __restrict__ fi, global {0}* __restrict__ pq0, {0} *overUnder, {0} *diag, {0} piLyambda, {0} subsupra2)
{{//using m + 1 in λ(m) because m is zero based(for arrays)
	{0} lyambda2 = lyambdaDiv2(m + 1, piLyambda, subsupra2);
	{0} muk2, muk1;
	Un(k - 2, lyambda2, &muk2, &muk1);//SNE p.234 bottom
	{0} muk0 = HP(2.0 * lyambda2 * muk1 - muk2);//Un(k - 0, lyambda2, sqrx);
	*overUnder = HP(One / (muk1 * muk1 - muk2 * muk2));//SNE p.235 below (37);error - in book used μₖ0 instead of muk1(see p.233(34))
	*diag = HP(muk1 * (muk0 - muk2) * overUnder[0]);//SNE p.235 below (37)

	for (int l = 0; l < L; l++)//SNE p.233(34)
	{{
		int lp = l;//indexing p array
		int lq = l + L;//indexing q array
		{0} fiplus = HP((pq0[lq * colsPQ] + pq0[lp * colsPQ]) / (2.0 * (muk2 - muk1)));//SNE p.235(37)
		{0} fiminu = HP((pq0[lq * colsPQ] - pq0[lp * colsPQ]) / (2.0 * (muk2 + muk1)));//SNE p.235(37)
		fi[2 * l + 1 - 1] = HP(fiplus + fiminu);//SNE p.235(37) ' - 1' for zero base
		fi[2 * l + 2 - 1] = HP(fiplus - fiminu);//SNE p.235(37) ' - 1' for zero base
	}}
}}

void calcAlpha(global {0} *alfa, {0} overUnder, {0} diag)
{{
	alfa[0] = HP(overDiag(0, overUnder) / diag);//SNR p.75(7)
	for (int i = 1; i <= alfaUB; i++) alfa[i] = HP(overDiag(i, overUnder) / (diag - underDiag(i, overUnder) * alfa[i - 1]));//SNR p.75(7)
}}

void progonka(global {0}* __restrict__ w, global {0}* __restrict__ fi, global {0}* __restrict__ alfa, {0} overUnder, {0} diag)
{{
	calcAlpha(alfa, overUnder, diag);

	//ŵ as beta(left part)
	w[0 * colsPQ] = HP(fi[0] / diag);//SNR p.77(12)
	for (int i = 1; i <= alfaUB; i++) w[i * colsPQ] = HP((fi[i] + underDiag(i, overUnder) * w[(i - 1) * colsPQ]) / (diag - underDiag(i, overUnder) * alfa[i - 1]));//SNR p.77(12,formula 2)

	//ŵ as beta(right part)
	int U = 2 * L - 1;//progonka upper bounds
	w[U * colsPQ] = HP(fi[U] / diag);//SNR p.77(12)
	for (int i = U - 1; i > alfaUB; i--) w[i * colsPQ] = HP((fi[i] + overDiag(i, overUnder) * w[(i + 1) * colsPQ]) / (diag - overDiag(i, overUnder) * alfa[(U - 1) - i]));//SNR p.77(12,formula 4)

	//ŵ as beta(at right side) & ŵ as ŵ(at left side)
	w[L * colsPQ] = HP((w[(alfaUB + 1) * colsPQ] + alfa[alfaUB] * w[alfaUB * colsPQ]) / (1.0 - alfa[alfaUB] * alfa[alfaUB]));//SNR p.77(13,formula 3)

	for (int i = L - 1; i >= 0; i--) w[i * colsPQ] = HP(w[i * colsPQ] + alfa[i] * w[(i + 1) * colsPQ]);//SNR p.77(13,formula 1)
	for (int i = L + 1; i <= U; i++) w[i * colsPQ] = HP(w[i * colsPQ] + alfa[U - i] * w[(i - 1) * colsPQ]);//SNR p.77(13,formula 2)
}}
", Utils.getTypeName<T>(), ArithmeticReplacer.convertTo_mul_HD<T>("piLyambda", "n"));

			string programSource = string.Format(
@"
{{
	int m = get_global_id(0);

	if (m < M)
	{{
		{0} overUnder, diag;
		int offsAl = m * L;
		int offsFi = offsAl << 1;
		int offsM = m + 1;//pq0 have 1 element padding after fft

		calculateFi(m, fi + offsFi, pq0 + offsM, &overUnder, &diag, piLyambda, subsupra2);
		progonka(pq0 + offsM, fi + offsFi, alfa + offsAl, overUnder, diag);
	}}
}}
", Utils.getTypeName<T>());
			string strDefines = string.Format(defines, L, k, M, pqDim2);
			string strProgram = strDefines + functions + strProgramHeader + programSource;
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + ArithmeticReplacer.replaceHPMacros(HighPrecisionOCL.strDD128Trig) + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + ArithmeticReplacer.replaceHPMacros(HighPrecisionOCL.strQD256Trig) + strProgram;
			}

			ProgramOCL program = UtilsCL.createProgram(strProgram, null, context, device);
			return program.CreateKernel(functionName);
		}
	}
}

using VLP2D.Common;

namespace VLP2D.Model
{
	internal static class MarchingKernelReducedSystemCU
	{
		public static string createProgramReducedSystem(string functionName, string strTypeName)
		{
			string args = "({0}* __restrict__ fi, {0}* __restrict__ alfa, {0}* __restrict__ pq0, {0} piLyambda, {0} subsupra2)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string functions =
@"
__device__ {0} lyambdaDiv2(int n, {0} piLyambda, {0} subsupra2)
{{
	{0} _sin = sin(piLyambda * n);
	return 1.0 + subsupra2 * _sin * _sin;//SNE p.236 under (39)
}}

__device__ void Un(int n, {0} x, {0} *out1, {0} *out2)
{{
	if (n == 0)
	{{
		*out1 = One;
		*out2 = 2.0 * x;
		return;
	}}
	if (n == 1)
	{{
		*out1 = 2.0 * x;
		*out2 = 4.0 * x * x - One;
		return;
	}}
	{0} u0 = Zero, u1 = One, u2 = 2.0 * x;
	for (int i = 0; i < n - 1; i++)
	{{
		u0 = u1;
		u1 = u2;
		u2 = 2.0 * x * u1 - u0;
	}}
	*out1 = u2;
	u0 = u1;
	u1 = u2;
	*out2 = 2.0 * x * u1 - u0;
}}

__device__ void calculateFi(int m, {0}* __restrict__ fi, {0}* __restrict__ pq0, {0} *overUnder, {0} *diag, {0} piLyambda, {0} subsupra2)
{{//using m + 1 in λ(m) because m is zero based(for arrays)
	{0} lyambda2 = lyambdaDiv2(m + 1, piLyambda, subsupra2);
	{0} muk2, muk1;
	Un(k - 2, lyambda2, &muk2, &muk1);//SNE p.234 bottom
	{0} muk0 = 2.0 * lyambda2 * muk1 - muk2;//Un(k - 0, lyambda2, sqrx);
	*overUnder = 1.0 / (muk1 * muk1 - muk2 * muk2);//SNE p.235 below (37);error - in book used μₖ0 instead of muk1(see p.233(34))
	*diag = muk1 * (muk0 - muk2) * (*overUnder);//SNE p.235 below (37)

	for (int l = 0; l < L; l++)//SNE p.233(34)
	{{
		int lp = l;//indexing p array
		int lq = l + L;//indexing q array
		{0} fiplus = (pq0[lq * colsPQ] + pq0[lp * colsPQ]) / (2.0 * (muk2 - muk1));//SNE p.235(37)
		{0} fiminu = (pq0[lq * colsPQ] - pq0[lp * colsPQ]) / (2.0 * (muk2 + muk1));//SNE p.235(37)
		fi[2 * l + 1 - 1] = fiplus + fiminu;//SNE p.235(37) ' - 1' for zero base
		fi[2 * l + 2 - 1] = fiplus - fiminu;//SNE p.235(37) ' - 1' for zero base
	}}
}}

__device__ void calcAlpha({0} *alfa, {0} overUnder, {0} diag)
{{
	alfa[0] = overDiag(0, overUnder) / diag;//SNR p.75(7)
	for (int i = 1; i <= alfaUB; i++) alfa[i] = overDiag(i, overUnder) / (diag - underDiag(i, overUnder) * alfa[i - 1]);//SNR p.75(7)
}}

__device__ void progonka({0}* __restrict__ w, {0}* __restrict__ fi, {0}* __restrict__ alfa, {0} overUnder, {0} diag)
{{
	calcAlpha(alfa, overUnder, diag);

	//ŵ as beta(left part)
	w[0 * colsPQ] = fi[0] / diag;//SNR p.77(12)
	for (int i = 1; i <= alfaUB; i++) w[i * colsPQ] = (fi[i] + underDiag(i, overUnder) * w[(i - 1) * colsPQ]) / (diag - underDiag(i, overUnder) * alfa[i - 1]);//SNR p.77(12,formula 2)

	//ŵ as beta(right part)
	int U = 2 * L - 1;//progonka upper bounds
	w[U * colsPQ] = fi[U] / diag;//SNR p.77(12)
	for (int i = U - 1; i > alfaUB; i--) w[i * colsPQ] = (fi[i] + overDiag(i, overUnder) * w[(i + 1) * colsPQ]) / (diag - overDiag(i, overUnder) * alfa[(U - 1) - i]);//SNR p.77(12,formula 4)

	//ŵ as beta(at right side) & ŵ as ŵ(at left side)
	w[L * colsPQ] = (w[(alfaUB + 1) * colsPQ] + alfa[alfaUB] * w[alfaUB * colsPQ]) / (1.0 - alfa[alfaUB] * alfa[alfaUB]);//SNR p.77(13,formula 3)

	for (int i = L - 1; i >= 0; i--) w[i * colsPQ] += alfa[i] * w[(i + 1) * colsPQ];//SNR p.77(13,formula 1)
	for (int i = L + 1; i <= U; i++) w[i * colsPQ] += alfa[U - i] * w[(i - 1) * colsPQ];//SNR p.77(13,formula 2)
}}";
			string programSource =
@"
{{
	int m = blockDim.x * blockIdx.x + threadIdx.x;

	if (m < M)
	{{
		{0} overUnder, diag;
		int offsAl = m * L;
		int offsFi = offsAl << 1;
		int offsM = m + 1;//pq0 have 1 element padding after fft

		calculateFi(m, fi + offsFi, pq0 + offsM, &overUnder, &diag, piLyambda, subsupra2);
		progonka(pq0 + offsM, fi + offsFi, alfa + offsAl, overUnder, diag);
	}}
}}";
			return string.Format(functions + strProgramHeader + programSource, strTypeName);
		}
	}
}

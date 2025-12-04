using ManagedCuda;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FFTLomontBaseCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		protected CudaDeviceVariable<T> cosTable, sinTable;
		protected CudaDeviceVariable<int> indJ;
		protected CudaKernel kernelReverse, kernelTableFFT, kernelRealFFT;
		protected SineTransformCU<T> sineTransform;
		T _2 = T.CreateTruncating(2);

		public FFTLomontBaseCU(CudaContext ctx, int fftSize)
		{
			sineTransform = new SineTransformCU<T>(ctx, fftSize);
			int fftSizeHalf = fftSize / 2;
			createSinCosTables(fftSizeHalf);
			indJ = createJIndesesCU(fftSizeHalf);
		}

		public string createProgramReverse(string functionName)
		{
			string args = "({0} *dataIn, int workSize, const int *indJ)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string function =
@"
//static __device__ __constant__ int indJ[fftSizeHalf / 4];//can be too big
__device__ void swap({0} *a, int idx1, int idx2)
{{
	{0} temp = a[idx1];
	a[idx1] = a[idx2];
	a[idx2] = temp;
}}
";
			string programSource =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < workSize)
	{{
		{0} *data = dataIn + idx * fftInOutSize;
		for (int k = 0; k < fftSizeHalf; k += 4)
		{{
			int j = indJ[k / 4];
			swap(data, j + 2, k + fftSizeHalf + 0);
			swap(data, j + 3, k + fftSizeHalf + 1);
			if (j > k)
			{{
				swap(data, j + 0, k + 0);
				swap(data, j + 1, k + 1);

				swap(data, j + fftSizeHalf + 2, k + fftSizeHalf + 2);
				swap(data, j + fftSizeHalf + 3, k + fftSizeHalf + 3);
			}}
		}} // bit reverse loop
	}}
}}";
			return string.Format(function + strProgramHeader + programSource, Utils.getTypeName<T>());
		}

		public string createProgramTableFFT(string functionName)
		{
			string args = "({0} *dataIn, int workSize, const {0} *sinTable, const {0} *cosTable)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSource =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < workSize)
	{{
		{0} *data = dataIn + idx * fftInOutSize;

		// do transform: so single point transforms, then doubles, etc.
		int mmax = 1;
		int tptr = 0;
		while (fftSizeHalf > mmax)
		{{
			int istep = 2 * mmax;
			for (int m = 0; m < istep; m += 2)
			{{
				{0} wr = cosTable[tptr];
				{0} wi = -sinTable[tptr++];
				for (int k = m; k < fftSize; k += 2 * istep)
				{{
					int j = k + istep;
					{0} tempr = wr * data[j + 0] - wi * data[j + 1];
					{0} tempi = wi * data[j + 0] + wr * data[j + 1];
					data[j + 0] = data[k + 0] - tempr;
					data[j + 1] = data[k + 1] - tempi;
					data[k + 0] += tempr;
					data[k + 1] += tempi;
				}}
			}}
			mmax = istep;
		}}
	}}
}}";
			return string.Format(strProgramHeader + programSource, Utils.getTypeName<T>());
		}

		public string createProgramRealFFT(string functionName)
		{
			string args = "({0} *dataIn, int workSize)";
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSource =
@"
{{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < workSize)
	{{
		{0} *data = dataIn + idx * fftInOutSize;
		{0} wjr = wpr;
		{0} wji = wpi;

		for (int j = 1; j <= fftSize / 4; ++j)
		{{
			int k = fftSizeHalf - j;
			{0} tkr = data[2 * k + 0];    // real and imaginary parts of t_k  = t_(n/2 - j)
			{0} tki = data[2 * k + 1];
			{0} tjr = data[2 * j + 0];    // real and imaginary parts of t_j
			{0} tji = data[2 * j + 1];

			{0} a = (tjr - tkr) * wji;
			{0} b = (tji + tki) * wjr;
			{0} c = (tjr - tkr) * wjr;
			{0} d = (tji + tki) * wji;
			{0} e = (tjr + tkr);
			{0} f = (tji - tki);

			// compute entry y[j]
			data[2 * j + 0] = (e + (a + b)) * 0.5;
			data[2 * j + 1] = (f + (d - c)) * 0.5;

			// compute entry y[k]
			data[2 * k + 0] = (e - (b + a)) * 0.5;
			data[2 * k + 1] = ((d - c) - f) * 0.5;

			{0} temp = wjr;
			wjr = wjr * wpr - wji * wpi;  
			wji = temp * wpi + wji * wpr;
		}}

		data[0] += data[1];
	}}
}}";
			return string.Format(strProgramHeader + programSource, Utils.getTypeName<T>());
		}

		public void cleanup()
		{
			UtilsCU.disposeBuf(ref cosTable);
			UtilsCU.disposeBuf(ref sinTable);
			UtilsCU.disposeBuf(ref indJ);

			sineTransform?.cleanup();
			sineTransform = null;
		}

		void createSinCosTables(int fftSizeHalf)
		{
			T[] cosTableTmp = new T[fftSizeHalf];
			T[] sinTableTmp = new T[fftSizeHalf];
			InitializeTables(fftSizeHalf, cosTableTmp, sinTableTmp);
			cosTable = cosTableTmp;
			sinTable = sinTableTmp;
		}

		void InitializeTables(int numPoints, T[] cosTable, T[] sinTable)
		{
			// forward pass
			int mmax = 1, pos = 0;
			while (numPoints > mmax)
			{
				int istep = 2 * mmax;
				T theta = T.Pi / T.CreateTruncating(mmax);
				T wr = T.One, wi = T.Zero;
				T wpi = T.Sin(theta);
				// compute in a slightly slower yet more accurate manner
				T wpr = T.Sin(theta / _2);
				wpr = -wpr * wpr * _2;
				for (int m = 0; m < istep; m += 2)
				{
					cosTable[pos] = wr;
					sinTable[pos++] = wi;
					T t = wr;
					wr = wr * wpr - wi * wpi + wr;
					wi = wi * wpr + t * wpi + wi;
				}
				mmax = istep;
			}
		}

		int[] createJIndesesCU(int fftSizeHalf)
		{
			int cnt = fftSizeHalf / 4;
			if (cnt == 0) cnt = 1;
			int[] indTmp = new int[cnt];
			createJIndeses(indTmp);
			return indTmp;
		}

		void createJIndeses(int[] indJ)
		{
			int j = 0;
			indJ[0] = 0;
			for (int k = 1; k < indJ.Length; k++)
			{
				// Knuth R4: advance j
				int h = indJ.Length * 2;// this is Knuth's 2^(n-1)
				while (j >= h)
				{
					j -= h;
					h /= 2;
				}
				j += h;
				indJ[k] = j;
			}
		}
	}
}
using Cloo;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FFTLomontBaseOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		protected BufferOCL<T> cosTable, sinTable;
		protected BufferOCL<int> indJ;
		protected KernelOCL kernelReverse, kernelTableFFT, kernelRealFFT;
		protected SineTransformOCL<T> sineTransform;

		public FFTLomontBaseOCL(CommandQueueOCL commands, int fftSize)
		{
			sineTransform = new SineTransformOCL<T>(commands);
			sineTransform.createKernelPreProcess(commands.Device, commands.Context, fftSize);
			sineTransform.createKernelPostProcess(commands.Device, commands.Context, fftSize);
			int fftSizeHalf = fftSize / 2;
			createSinCosTables(commands.Context, fftSizeHalf);
			createJIndesesCL(commands.Context, fftSizeHalf);
		}

		public string createProgramDefines(int fftSize)
		{
			string strDefines =
@"
#define fftSize			{0}
#define fftSizeHalf		{1}
#define fftOutputSize	{2}
";
			return string.Format(strDefines, fftSize, fftSize / 2, (fftSize / 2 + 1) * 2);//Hermitian redundancy
		}

		public string createProgramReverse(string functionName)
		{
			string args = "(global {0} *dataIn, int workSize, global const int *indJ)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string function =
@"
void swap(global {0} *a, int idx1, int idx2)
{{
	{0} temp = a[idx1];
	a[idx1] = a[idx2];
	a[idx2] = temp;
}}
";
			string programSource =
@"
{{
	int idx = get_global_id(0);

	if (idx < workSize)
	{{
		global {0} *data = dataIn + idx * fftOutputSize;
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
			string args = "(global {0} *dataIn, int workSize, global const {0} *sinTable, global const {0} *cosTable)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string programSource =
@"
{{
	int idx = get_global_id(0);

	if (idx < workSize)
	{{
		global {0} *data = dataIn + idx * fftOutputSize;

		// do transform: so single point transforms, then doubles, etc.
		int mmax = 1;
		int tptr = 0;
		while (fftSizeHalf > mmax)
		{{
			int istep = 2 * mmax;
			for (int m = 0; m < istep; m += 2)
			{{
				{0} wr = cosTable[tptr];
				{0} wi = HP(-sinTable[tptr++]);//negation
				for (int k = m; k < fftSize; k += 2 * istep)
				{{
					int j = k + istep;
					{0} tempr = HP(wr * data[j + 0] - wi * data[j + 1]);
					{0} tempi = HP(wi * data[j + 0] + wr * data[j + 1]);
					data[j + 0] = HP(data[k + 0] - tempr);
					data[j + 1] = HP(data[k + 1] - tempi);
					data[k + 0] = HP(data[k + 0] + tempr);
					data[k + 1] = HP(data[k + 1] + tempi);
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
			string args = "(global {0} *dataIn, int workSize, const {0} wpr, const {0} wpi)";
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string programSource =
@"
{{
	int idx = get_global_id(0);

	if (idx < workSize)
	{{
		global {0} *data = dataIn + idx * fftOutputSize;
		{0} wjr = wpr;
		{0} wji = wpi;

		for (int j = 1; j <= fftSize / 4; ++j)
		{{
			int k = fftSizeHalf - j;
			{0} tkr = data[2 * k + 0];    // real and imaginary parts of t_k  = t_(n/2 - j)
			{0} tki = data[2 * k + 1];
			{0} tjr = data[2 * j + 0];    // real and imaginary parts of t_j
			{0} tji = data[2 * j + 1];

			{0} a = HP((tjr - tkr) * wji);
			{0} b = HP((tji + tki) * wjr);
			{0} c = HP((tjr - tkr) * wjr);
			{0} d = HP((tji + tki) * wji);
			{0} e = HP(tjr + tkr);
			{0} f = HP(tji - tki);

			// compute entry y[j]
			{0} ab = HP(a + b);
			{0} dc = HP(d - c);
			data[2 * j + 0] = HP((e + ab) * 0.5);
			data[2 * j + 1] = HP((f + dc) * 0.5);

			// compute entry y[k]
			data[2 * k + 0] = HP((e - ab) * 0.5);
			data[2 * k + 1] = HP((dc - f) * 0.5);

			{0} temp = wjr;
			wjr = HP(wjr * wpr - wji * wpi);
			wji = HP(temp * wpi + wji * wpr);
		}}

		data[0] = HP(data[0] + data[1]);
	}}
}}";
			return string.Format(strProgramHeader + programSource, Utils.getTypeName<T>());
		}

		public void cleanup()
		{
			UtilsCL.disposeBuf(ref cosTable);
			UtilsCL.disposeBuf(ref sinTable);
			UtilsCL.disposeBuf(ref indJ);

			sineTransform?.cleanup();
			sineTransform = null;
		}

		void createSinCosTables(ContextOCL ctx, int fftSizeHalf)
		{
			T[] cosTableTmp = new T[fftSizeHalf];
			T[] sinTableTmp = new T[fftSizeHalf];
			InitializeTables(fftSizeHalf, cosTableTmp, sinTableTmp);
			cosTable = new BufferOCL<T>(ctx, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, cosTableTmp);
			sinTable = new BufferOCL<T>(ctx, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, sinTableTmp);
		}

		void InitializeTables(int numPoints, T[] cosTable, T[] sinTable)
		{
			// forward pass
			T _2 = T.CreateTruncating(2);
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

		void createJIndesesCL(ContextOCL ctx, int fftSizeHalf)
		{
			int cnt = fftSizeHalf / 4;
			if (cnt == 0) cnt = 1;
			int[] indTmp = new int[cnt];
			createJIndeses(indTmp);
			indJ = new BufferOCL<int>(ctx, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, indTmp);
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
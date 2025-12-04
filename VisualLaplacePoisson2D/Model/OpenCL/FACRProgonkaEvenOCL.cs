
using Cloo;
using DD128Numeric;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class FACRProgonkaEvenOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		CommandQueueOCL commands;
		KernelOCL kernelProgonka, kernelAlfaCounts;
		long[] gWorkSize = { 0 };
		BufferOCL<T> alpha;
		BufferOCL<int> alfaOffsets, alfaCounts;
		int allProgonkaWorkSize;

		public FACRProgonkaEvenOCL(CommandQueueOCL commands, BufferOCL<T> un, int allProgonkaWorkSize, int progonkaSize, int N2, int paramL, T stepX2, T stepY2)
		{
			if ((progonkaSize & 1) == 1) throw new System.Exception("FACRProgonkaEvenOCL progonkaSize should be even");
			this.commands = commands;
			this.allProgonkaWorkSize = allProgonkaWorkSize;

			T step2DivStep2 = stepX2 / stepY2;
			createKernelMeetingProgonkaEven(un, progonkaSize, N2 - 1, paramL);
			gWorkSize[0] = allProgonkaWorkSize;

			alfaCounts = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadWrite, allProgonkaWorkSize);

			T _2 = T.CreateTruncating(2);
			T x2DivY2Doubled = step2DivStep2 * _2;
			object[] argsProgonka = [un, null/*alpha*/, null/*alfaOffsets*/, alfaCounts, AlfaСonvergentUpperBound<T>.get_2tm1(), stepX2, step2DivStep2, x2DivY2Doubled, T.Pi / T.CreateTruncating(N2), x2DivY2Doubled + _2];
			UtilsCL.setKernelArguments<T>(kernelProgonka, argsProgonka);

			object[] argsAlfaCounts = [alfaCounts, AlfaСonvergentUpperBound<T>.get_2tm1(), x2DivY2Doubled, T.Pi / T.CreateTruncating(N2), x2DivY2Doubled + _2];
			UtilsCL.setKernelArguments<T>(kernelAlfaCounts, argsAlfaCounts);
		}

		public void calculate()
		{
			commands.Execute(kernelAlfaCounts, null, gWorkSize, null, null);
			commands.Finish();

			int size = allProgonkaWorkSize;
			int[] alfaCountsCPU = new int[allProgonkaWorkSize];
			commands.ReadFromBuffer(alfaCounts, ref alfaCountsCPU, true, null);
			int[] offsets = new int[size];
			int offset = 0;
			for (int i = 0; i < size; i++)
			{
				offsets[i] = offset;
				offset += alfaCountsCPU[i];
			}

			alpha = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, offset);
			alfaOffsets = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, offsets);

			kernelProgonka.SetMemoryArgument(1, alpha);
			kernelProgonka.SetMemoryArgument(2, alfaOffsets);

			commands.Execute(kernelProgonka, null, gWorkSize, null, null);
			commands.Finish();
		}

		public void cleanup()
		{
			UtilsCL.disposeBuf(ref alfaOffsets);
			UtilsCL.disposeBuf(ref alfaCounts);
			UtilsCL.disposeBuf(ref alpha);
			commands = null;
			UtilsCL.disposeKP(ref kernelProgonka);
		}

		void createKernelMeetingProgonkaEven(BufferOCL<T> un, int progonkaSize, int dimY, int paramL)
		{
			string typeName = Utils.getTypeName<T>();
			string definesProgonka =
@"
#define U		{0}//upper bound
#define midX	{1}
#define dimY	{2}
#define paramL	{3}
#define countDiagElems	{4}
#define identity(a) (a)//workaround for substitutor

#define ind(i) (((i) < alfaCount) ? (i) : alfaCount - 1)
";
			string strFunctions = string.Format(
@"
int alfaConvergentCount(float diagElem, int _2tm1)
{{
	float u = (diagElem + sqrt(diagElem * diagElem - 4)) / 2;
	float K = (float)_2tm1 / log2(u);
	return min(midX, (int)K);
}}

void calcAlfa({0} diagElem, global {0} *alfa, int alfaCount)
{{
	alfa[0] = HP(1.0 / diagElem);//[SNR] p.75(7)
	for (int i = 1; i < alfaCount; i++) alfa[i] = HP(1.0 / (diagElem - alfa[i - 1]));//[SNR] p.75(7)
}}

void progonka(global {0} *un, int j, global {0} *alfa, {0} coef, int alfaCount)
{{
	//calc beta, put to un
	for (int i = 0; i <= midX - 1; i++)
	{{
		un[(i + 0) * dimY + j] = HP((coef * un[(i + 0) * dimY + j] + identity(i != 0 ? un[(i - 1) * dimY + j] : Zero)) * alfa[ind(i)]);
		un[(U - i) * dimY + j] = HP((coef * un[(U - i) * dimY + j] + identity(i != 0 ? un[(U - (i - 1)) * dimY + j] : Zero)) * alfa[ind(i)]);
	}}

	un[midX * dimY + j] = HP((un[midX * dimY + j] + alfa[ind(midX - 1)] * un[(midX - 1) * dimY + j]) / (1.0 - alfa[ind(midX - 1)] * alfa[ind(midX - 1)]));//[SNR] p.77(bottom)

	//from middle to left, then from middle to right
	for (int i = midX - 1,i1 = i * dimY; i >= 0; i--,i1 -= dimY) un[i1 + j] = HP(un[i1 + j] + alfa[ind(i + 0)] * un[i1 + dimY + j]);
	for (int i = midX + 1,i1 = i * dimY; i <= U; i++,i1 += dimY) un[i1 + j] = HP(un[i1 + j] + alfa[ind(U - i)] * un[i1 - dimY + j]);//[SNR] p.75(7)
}}
", typeName);
			string functionNameProgonkaEven = "progonkaMeetingM2Even";
			string srcProgonka =
	@"
(global {0} *un, global {0} *alfaIn, global int *alfaOffsets, global int *alfaCounts, int _2tm1, {0} stepX2, {0} x2DivY2, {0} x2DivY2Doubled, {0} piDivN2, {0} ck2base)
{{
	int j0 = get_global_id(0);

	global {0} *alfa = alfaIn + alfaOffsets[j0];//j0 * (midX + 1);

	{0} PiJDivN2 = {1};
	{0} cosinus = cos(PiJDivN2);
	{0} diagDelta = HP(x2DivY2Doubled * cosinus);
	int j = ((j0 + 1) << paramL) - 1;//even in CPU variant(because of boundaries)

	calcAlfa(HP(ck2base - diagDelta), alfa, alfaCounts[j0]);
	progonka(un, j, alfa, stepX2, alfaCounts[j0]);

	#if paramL >= 1
	{0} diagElems[countDiagElems];
	diagElems[0] = HP(ck2base + diagDelta);
	#if paramL >= 2
	{0} sinus = sin(PiJDivN2);
	diagElems[1] = HP(-x2DivY2Doubled * sinus);
	diagElems[2] = HP(ck2base - diagElems[1]);
	diagElems[1] = HP(diagElems[1] + ck2base);
	#endif
	#if paramL >= 3
	{0} arg = HP(PiJDivN2 - Pi4);
	diagElems[3] = HP(-x2DivY2Doubled * cos(arg));
	diagElems[4] = HP(ck2base - diagElems[3]);
	diagElems[3] = HP(diagElems[3] + ck2base);

	diagElems[5] = HP(-x2DivY2Doubled * sin(arg));
	diagElems[6] = HP(ck2base - diagElems[5]);
	diagElems[5] = HP(diagElems[5] + ck2base);
	#endif

	for (int k = 0; k < countDiagElems; k++)
	{{
		int alfaCount = alfaConvergentCount(to_float(diagElems[k]), _2tm1);
		calcAlfa(diagElems[k], alfa, alfaCount);
		progonka(un, j, alfa, x2DivY2, alfaCount);
	}}
	#endif
}}
";
			string functionNameAlfaLengths = "alfaLengths";
			string argsAlfaLengths = "(global int *alfaCounts, int _2tm1, {0} x2DivY2Doubled, {0} piDivN2, {0} ck2base)";
			string srcAlfaLengths =
@"
{{
	int j = get_global_id(0);

	float PiJDivN2 = (j + 1) * to_float(piDivN2);
	float _ck2base = to_float(ck2base);
	float _x2DivY2Doubled = to_float(x2DivY2Doubled);
	float minDiag = _ck2base - _x2DivY2Doubled * cos(PiJDivN2);

	if (paramL >= 2)
	{{
		minDiag = min(minDiag, _ck2base - _x2DivY2Doubled * sin(PiJDivN2));
	}}
	if (paramL >= 3)
	{{
		float arg = PiJDivN2 - to_float(Pi4);
		float diagDelta = _x2DivY2Doubled * cos(arg);
		minDiag = min(minDiag, _ck2base - diagDelta);
		minDiag = min(minDiag, _ck2base + diagDelta);

		diagDelta = _x2DivY2Doubled * sin(arg);
		minDiag = min(minDiag, _ck2base - diagDelta);
		minDiag = min(minDiag, _ck2base + diagDelta);
	}}

	alfaCounts[j] = alfaConvergentCount(minDiag, _2tm1);
}}
";
			string defines = string.Format(definesProgonka, progonkaSize - 1, progonkaSize / 2, dimY, paramL, (1 << paramL) - 1);
			string strProgonka = string.Format(srcProgonka, typeName, ArithmeticReplacer.convertTo_mul_HD<T>("piDivN2", "(j0 + 1)"));
			string strProgram = defines + strFunctions + UtilsCL.kernelPrefix + functionNameProgonkaEven + strProgonka;
			strProgram += string.Format(UtilsCL.kernelPrefix + functionNameAlfaLengths + argsAlfaLengths + srcAlfaLengths, typeName);
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else
			{
				strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + ArithmeticReplacer.replaceHPMacros(HighPrecisionOCL.strDD128Trig) + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + ArithmeticReplacer.replaceHPMacros(HighPrecisionOCL.strQD256Trig) + strProgram;
			}

			string compileOptions = typeof(T) == typeof(QD256) ? "-cl-opt-disable" : null;//to not cause InvalidCommandQueueException
			ProgramOCL program = UtilsCL.createProgram(strProgram, compileOptions, commands.Context, commands.Device);
			kernelProgonka = program.CreateKernel(functionNameProgonkaEven);
			kernelAlfaCounts = program.CreateKernel(functionNameAlfaLengths);
		}
	}
}

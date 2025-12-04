using Cloo;
using DD128Numeric;
using QD256Numeric;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class ProgonkaSchemeOCL<T> : Iterative1DScheme<T>, IScheme<T> where T : struct, INumber<T>, IRootFunctions<T>, IPowerFunctions<T>, IMinMaxValue<T>
	{
		protected T stepX2, stepY2, dt;
		protected T[] alphaX, alphaY;//beta is placed to result
		protected int cXSegments, cYSegments;
		protected readonly bool bProgonkaFixedIters;
		protected T eps;
		protected Action<int> calculateIterationAlpha = null;

		protected CommandQueueOCL commands;
		protected KernelOCL[] kernels;
		KernelOCL kernelEpsExceeded;
		protected BufferOCL<T> unOCL0, unOCL1, unOCLm, fn;
		protected BufferOCL<T> alphaXOCL, alphaYOCL;
		protected T[] un;
		protected BufferOCL<int> flagOCL;
		int[] flag = { 0 };
		long[] gWork1DOffset = new long[] { 1 };
		long[] gWork2DOffset = new long[] { 1, 1 };
		long[] workSizeX, workSizeY, workSize2DInternalPoints;
		bool uuChanged = false;
		protected T _2 = T.CreateTruncating(2);
		protected readonly string strDefinesProgonkaX,strDefinesProgonkaY;
		protected readonly string definesProgonkaX =
@"
#define dimX	{0}
#define dimY	{1}
";

		protected string programSourceProgonkaX =
@"
{{
	int j = get_global_id(0);//indeces are 1-based, workgroup indeces are 1-based

	for (int i = 1,i1 = dimY; i < dimX - 1; i++,i1 += dimY) unDst[i1 + j] = HP(alphaX[i] * ({0} + unDst[i1 - dimY + j]));//unSrc is used inside strRightSideX
	for (int i = dimX - 2,i1 = dimY * (dimX - 2); i > 0; i--,i1 -= dimY) unDst[i1 + j] = HP(unDst[i1 + j] + alphaX[i] * unDst[i1 + dimY + j]);
}}";

		protected readonly string definesProgonkaY =
@"
#define dimY	{0}
";

		protected string programSourceProgonkaY =
@"
{{
	int i = get_global_id(0);//indeces are 1-based, workgroup indeces are 1-based

	i *= dimY;
	for (int j = 1; j < dimY - 1; j++) unDst[i + j] = HP(alphaY[j] * ({0} + unDst[i + j - 1]));//unSrc is used inside strRightSideY
	for (int j = dimY - 2; j > 0; j--) unDst[i + j] = HP(unDst[i + j] + alphaY[j] * unDst[i + j + 1]);
}}";

		public ProgonkaSchemeOCL(int cXSegments1, int cYSegments1, T stepX, T stepY, Func<T, T, T> fKsi, T epsIn, PlatformOCL platform, DeviceOCL device, bool bProgonkaFixedIters)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			cXSegments = cXSegments1;
			cYSegments = cYSegments1;

			dimX = cXSegments + 1;
			dimY = cYSegments + 1;

			un = new T[dimX * dimY];

			alphaX = new T[cXSegments];
			alphaY = new T[cYSegments];

			eps = epsIn;

			commands = UtilsCL.createCommandQueue(platform, device, CommandQueueFlagsOCL.None);

			unOCL0 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dimX * dimY);
			unOCL1 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dimX * dimY);
			unOCLm = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dimX * dimY);

			if (fKsi != null)
			{
				GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => un[i * dimY + j] = fKsi(stepX * T.CreateTruncating(i), stepY * T.CreateTruncating(j)));
				fn = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, un);
			}

			flagOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadWrite, 1);

			kernels = new KernelOCL[2];

			workSizeX = new long[] { dimX - 2 };
			workSizeY = new long[] { dimY - 2 };
			workSize2DInternalPoints = new long[] { workSizeX[0], workSizeY[0] };

			this.bProgonkaFixedIters = bProgonkaFixedIters;
			if (!bProgonkaFixedIters) kernelEpsExceeded = UtilsCL.createProgramEpsExceeded(commands.Context, commands.Device, unOCL0, unOCL1, flagOCL, dimY, eps);

			calculateOptimalTimeStep(stepX, stepY);
			strDefinesProgonkaX = string.Format(definesProgonkaX, dimX, dimY);
			strDefinesProgonkaY = string.Format(definesProgonkaY, dimY);
		}

		public T doIteration(int iter)
		{
			uuChanged = true;
			calculateIterationAlpha?.Invoke(iter);

			setKernel0Arguments(iter);
			commands.Execute(kernels[0], gWork1DOffset, workSizeY, null, null);
			commands.Finish();

			setKernel1Arguments(iter);
			commands.Execute(kernels[1], gWork1DOffset, workSizeX, null, null);
			commands.Finish();

			T rc = T.Zero;
			if (bProgonkaFixedIters) rc = eps + eps;
			else rc = epsExceeded() ? eps + eps : T.Zero;

			UtilsSwap.swap(ref unOCL0, ref unOCL1);

			return rc;
		}

		protected virtual void setKernel0Arguments(int iter)
		{
			kernels[0].SetMemoryArgument(0, unOCL0);
		}

		protected virtual void setKernel1Arguments(int iter)
		{
			kernels[1].SetMemoryArgument(1, unOCL1);
		}

		protected void calcAlpha(T bx, T by)
		{
			alphaX[0] = T.Zero;
			for (int i = 1; i < cXSegments; i++) alphaX[i] = T.One / (bx - alphaX[i - 1]);//ax is 1.0(coeffs as in 426.xps - 1 is ax,coeff with alphaX is 1)

			alphaY[0] = T.Zero;
			for (int i = 1; i < cYSegments; i++) alphaY[i] = T.One / (by - alphaY[i - 1]);//ay is 1.0(coeffs as in 426.xps - 1 is ax,coeff with alphaX is 1)

			alphaXOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, alphaX);
			alphaYOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, alphaY);
		}

		bool epsExceeded()
		{
			flag[0] = 0;
			commands.WriteToBuffer(flag, flagOCL, true, null);

			commands.Execute(kernelEpsExceeded, gWork2DOffset, workSize2DInternalPoints, null, null);

			commands.ReadFromBuffer(flagOCL, ref flag, true, null);
			return flag[0] == 1;
		}

		public override T[] getArray()
		{
			if (uuChanged)
			{
				uuChanged = false;

				commands.ReadFromBuffer(unOCL0, ref un, true, null);
			}

			return un;
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
			commands.WriteToBuffer(un, unOCL0, true, null);
			commands.CopyBuffer(unOCL0, unOCL1, null);
			commands.CopyBuffer(unOCL0, unOCLm, null);
		}

		public virtual int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public virtual void cleanup()
		{
			UtilsCL.disposeQC(ref commands);
			UtilsCL.disposeKP(ref kernels[0]);
			UtilsCL.disposeKP(ref kernels[1]);
			UtilsCL.disposeKP(ref kernelEpsExceeded);
			UtilsCL.disposeBuf(ref unOCL0);
			UtilsCL.disposeBuf(ref unOCL1);
			UtilsCL.disposeBuf(ref unOCLm);
			UtilsCL.disposeBuf(ref fn);
			UtilsCL.disposeBuf(ref alphaXOCL);
			UtilsCL.disposeBuf(ref alphaYOCL);
			UtilsCL.disposeBuf(ref flagOCL);
		}

		protected ProgramOCL createProgram(string strProgram)
		{
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			//string compileOptions = typeof(T) == typeof(QD256) ? "-cl-opt-disable" : null;
			string compileOptions = null;//doesn't cause InvalidCommandQueueException

			return UtilsCL.createProgram(strProgram, compileOptions, commands.Context, commands.Device);//"-cl-opt-disable"
		}

		void calculateOptimalTimeStep(T stepX, T stepY)
		{
			T a = stepX * T.CreateTruncating(cXSegments);
			T b = stepY * T.CreateTruncating(cYSegments);
			T a2 = a * a;
			T b2 = b * b;
			//http://scask.ru/q_book_dig_m.php?id=177 p392 - search - шаг по времени счет на установление
			dt = T.Sqrt(stepX2 + stepY2) * T.Pow(T.One / a2 + T.One / b2, T.CreateTruncating(-0.5)) / T.Pi;//Kalitkin p.406, my value of dt was 0.001;
		}
	}
}

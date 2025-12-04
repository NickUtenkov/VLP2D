using Cloo;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VarDirSchemeOCL<T> : ProgonkaSchemeOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IPowerFunctions<T>, IMinMaxValue<T>
	{
		JordanSpeedup<T> jrd;
		bool equalSteps;

		public VarDirSchemeOCL(int cXSegments, int cYSegments, T stepX, T stepY, T eps, Func<T, T, T> fKsi, bool isJordan, PlatformOCL platform, DeviceOCL device) :
			base(cXSegments, cYSegments, stepX, stepY, fKsi, eps, platform, device, isJordan)
		{
			equalSteps = T.Abs(stepX - stepY) < T.Min(stepX, stepY) / T.CreateTruncating(100);

			T ω1 = T.Zero, ω2 = T.Zero;

			if (!isJordan)
			{
				ω1 = stepX2 * _2 / dt;
				ω2 = stepY2 * _2 / dt;
				calcAlpha(ω1 + _2, ω2 + _2);
			}
			else
			{
				jrd = new JordanSpeedup<T>(cXSegments, cYSegments, stepX2, stepY2, eps);

				calculateIterationAlpha = calcVariableDirectionsMethodAlpha;

				alphaXOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, alphaX.Length);
				alphaYOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, alphaY.Length);
			}

			createProgramProgonkaX(ω1);
			createProgramProgonkaY(ω2);
		}

		protected override void setKernel0Arguments(int iter)
		{
			base.setKernel0Arguments(iter);
			if (jrd != null) kernels[0].SetValueArgument(5, stepX2 * jrd.w1(iter));
		}

		protected override void setKernel1Arguments(int iter)
		{
			base.setKernel1Arguments(iter);
			if (jrd != null) kernels[1].SetValueArgument(5, stepY2 * jrd.w2(iter));
		}

		public override int maxIterations() { return (jrd != null) ? jrd.maxIters : 0; }

		public override IterationsKind iterationsKind()
		{
			return (jrd != null) ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}

		void calcVariableDirectionsMethodAlpha(int iter)
		{
			alphaX[0] = T.Zero;
			T w1kPlus2 = stepX2 * jrd.w1(iter) + _2;
			for (int i = 1; i < cXSegments; i++) alphaX[i] = T.One / (w1kPlus2 - alphaX[i - 1]);//[SNR] p.443, top

			alphaY[0] = T.Zero;
			T w2kPlus2 = stepY2 * jrd.w2(iter) + _2;
			for (int i = 1; i < cYSegments; i++) alphaY[i] = T.One / (w2kPlus2 - alphaY[i - 1]);

			commands.WriteToBuffer(alphaX, alphaXOCL, true, null);
			commands.WriteToBuffer(alphaY, alphaYOCL, true, null);
		}

		void createProgramProgonkaX(T ω1)
		{
			string functionName = "ProgonkaX";
			string args = "(global {0} *unSrc, global {0} *unDst, global {0} *alphaX, {0} stepX2, {0} stepX2DivY2, {0} srcCoefX" + (fn != null ? ", global {0} *fn)" : ")");
			args = string.Format(args, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string unMult = "(unSrc[i1 + j] * srcCoefX)";
			string operatorLyy = "(unSrc[i1 + (j - 1)] - 2.0 * unSrc[i1 + j] + unSrc[i1 + (j + 1)])";

			string term1 = equalSteps ? operatorLyy : string.Format("({0} * {1})", operatorLyy, "stepX2DivY2");
			string strRightSideX = string.Format("({0} + {1})", unMult, term1);
			if (fn != null) strRightSideX = string.Format("({0} + {1})", strRightSideX, "fn[i1 + j] * stepX2");

			string strProgram = strDefinesProgonkaX + strProgramHeader + String.Format(programSourceProgonkaX, strRightSideX);

			ProgramOCL program = createProgram(strProgram);
			kernels[0] = program.CreateKernel(functionName);

			kernels[0].SetMemoryArgument(0, unOCL0);
			kernels[0].SetMemoryArgument(1, unOCLm);
			kernels[0].SetMemoryArgument(2, alphaXOCL);
			kernels[0].SetValueArgument(3, stepX2);
			kernels[0].SetValueArgument(4, (stepX2 / stepY2));
			if (jrd == null) kernels[0].SetValueArgument(5, ω1);//else == stepX2 * jrd.w1(iter) on each iter
			if (fn != null) kernels[0].SetMemoryArgument(6, fn);
		}

		void createProgramProgonkaY(T ω2)
		{
			string functionName = "ProgonkaY";
			string args = "(global {0} *unSrc, global {0} *unDst, global {0} *alphaY, {0} stepY2, {0} stepY2DivX2, {0} srcCoefY" + (fn != null ? ", global {0} *fn)" : ")");
			args = string.Format(args, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string unMult = "(unSrc[i + j] * srcCoefY)";
			string operatorLxx = "(unSrc[(i - dimY) + j] - 2.0 * unSrc[i + j] + unSrc[(i + dimY) + j])";

			string term1 = equalSteps ? operatorLxx : string.Format("({0} * {1})", operatorLxx, "stepY2DivX2");
			string strRightSideY = string.Format("({0} + {1})", unMult, term1);
			if (fn != null) strRightSideY = string.Format("({0} + {1})", strRightSideY, "fn[i + j] * stepY2");

			string strProgram = strDefinesProgonkaY + strProgramHeader + String.Format(programSourceProgonkaY, strRightSideY);

			ProgramOCL program = createProgram(strProgram);
			kernels[1] = program.CreateKernel(functionName);

			kernels[1].SetMemoryArgument(0, unOCLm);
			kernels[1].SetMemoryArgument(1, unOCL1);
			kernels[1].SetMemoryArgument(2, alphaYOCL);
			kernels[1].SetValueArgument(3, stepY2);
			kernels[1].SetValueArgument(4, (stepY2 / stepX2));
			if (jrd == null) kernels[1].SetValueArgument(5, ω2);//else == stepY2 * jrd.w2(iter) on each iter
			if (fn != null) kernels[1].SetMemoryArgument(6, fn);
		}
	}
}

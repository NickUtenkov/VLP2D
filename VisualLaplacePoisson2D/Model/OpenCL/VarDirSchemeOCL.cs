using Cloo;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VarDirSchemeOCL<T> : ProgonkaSchemeOCL<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IPowerFunctions<T>, IMinMaxValue<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>
	{
		JordanSpeedup<T> jrd;
		bool equalSteps;

		public VarDirSchemeOCL(RectangleData<T> rectData, T eps, Func<T, T, T> fKsi, bool isJordan, PlatformOCL platform, DeviceOCL device) :
			base(rectData, fKsi, eps, platform, device, isJordan)
		{
			equalSteps = T.Abs(stepX - stepY) < T.Min(stepX, stepY) / T.CreateTruncating(100);

			T ω = T.Zero;

			if (!isJordan)
			{
				ω = _2 / dt;
				calcAlpha(_2 + stepX2 * ω, _2 + stepY2 * ω);
			}
			else
			{
				jrd = new JordanSpeedup<T>(cXSegments, cYSegments, stepX2, stepY2, eps);

				calculateIterationAlpha = calcVariableDirectionsMethodAlpha;
			}

			createProgramProgonkaX(ω);
			createProgramProgonkaY(ω);
		}

		protected override void setKernel0Arguments(int iter)
		{
			base.setKernel0Arguments(iter);
			if (jrd != null) kernels[0].SetValueArgument(5, jrd.w1(iter));
		}

		protected override void setKernel1Arguments(int iter)
		{
			base.setKernel1Arguments(iter);
			if (jrd != null) kernels[1].SetValueArgument(5, jrd.w2(iter));
		}

		public override int maxIterations() { return (jrd != null) ? jrd.maxIters : 0; }

		public override IterationsKind iterationsKind()
		{
			return (jrd != null) ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}

		void calcVariableDirectionsMethodAlpha(int iter) => calcAlpha(_2 + stepX2 * jrd.w1(iter), _2 + stepY2 * jrd.w2(iter));

		void createProgramProgonkaX(T ω)
		{
			string functionName = "ProgonkaX";
			string args = "(global {0} *src, global {0} *dst, global {0} *alphaX, {0} stepX2, {0} oneDivY2, {0} coef" + (fn != null ? ", global {0} *fn)" : ")");
			args = string.Format(args, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;

			string strRightSideX = string.Format("stepX2 * (src[i1 + j] * coef + (src[i1 + (j - 1)] - 2.0 * src[i1 + j] + src[i1 + (j + 1)]) * oneDivY2 + {0})", fn != null ? "fn[i1 + j]" : "0");
			string strProgram = strDefinesProgonkaX + strProgramHeader + String.Format(programSourceProgonkaX, strRightSideX);

			ProgramOCL program = createProgram(strProgram);
			kernels[0] = program.CreateKernel(functionName);

			kernels[0].SetMemoryArgument(0, unOCL0);
			kernels[0].SetMemoryArgument(1, unOCLm);
			kernels[0].SetMemoryArgument(2, alphaXOCL);
			kernels[0].SetValueArgument(3, stepX2);
			kernels[0].SetValueArgument(4, T.One / stepY2);
			if (jrd == null) kernels[0].SetValueArgument(5, ω);//else == jrd.w1(iter) on each iter
			if (fn != null) kernels[0].SetMemoryArgument(6, fn);
		}

		void createProgramProgonkaY(T ω)
		{
			string functionName = "ProgonkaY";
			string args = "(global {0} *src, global {0} *dst, global {0} *alphaY, {0} stepY2, {0} oneDivX2, {0} coef" + (fn != null ? ", global {0} *fn)" : ")");
			args = string.Format(args, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;

			string strRightSideY = string.Format("stepY2 * (src[i + j] * coef + (src[(i - dimY) + j] - 2.0 * src[i + j] + src[(i + dimY) + j]) * oneDivX2 + {0})", fn != null ? "fn[i + j]" : "0");
			string strProgram = strDefinesProgonkaY + strProgramHeader + String.Format(programSourceProgonkaY, strRightSideY);

			ProgramOCL program = createProgram(strProgram);
			kernels[1] = program.CreateKernel(functionName);

			kernels[1].SetMemoryArgument(0, unOCLm);
			kernels[1].SetMemoryArgument(1, unOCL1);
			kernels[1].SetMemoryArgument(2, alphaYOCL);
			kernels[1].SetValueArgument(3, stepY2);
			kernels[1].SetValueArgument(4, T.One / stepX2);
			if (jrd == null) kernels[1].SetValueArgument(5, ω);//else == jrd.w2(iter) on each iter
			if (fn != null) kernels[1].SetMemoryArgument(6, fn);
		}
	}
}

using Cloo;
using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class SplittingSchemeOCL<T> : ProgonkaSchemeOCL<T> where T : struct, INumber<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>
	{
		public SplittingSchemeOCL(int cXSegments, int cYSegments, T stepX, T stepY, T eps, Func<T, T, T> fKsi, PlatformOCL platform, DeviceOCL device) :
			base(cXSegments, cYSegments, stepX, stepY, fKsi, eps, platform, device, false)
		{//[S_VVCM] p.258 (27) σ₁,σ₂
			T σ1 = T.CreateTruncating(0.5);
			T σ2 = T.CreateTruncating(0.5);
			T diagExtraX = stepX2 / (dt * σ1);
			T diagExtraY = stepY2 / (dt * σ2);

			calcAlpha(_2 + diagExtraX, _2 + diagExtraY);

			createProgramProgonkaX(diagExtraX, (T.One - σ1) / σ1, stepX2 / (σ1 * _2));
			createProgramProgonkaY(diagExtraY, (T.One - σ2) / σ2, stepY2 / (σ2 * _2));
		}

		void createProgramProgonkaX(T srcCoefX, T operatorLxxCoef, T fnCoefX)
		{
			string functionName = "ProgonkaX";
			string args = "(global {0} *unSrc, global {0} *unDst, global {0} *alphaX, {0} srcCoefX, {0} operatorLxxCoef, {0} fnCoefX" + (fn != null ? ", global {0} *fn)" : ")");
			args = string.Format(args, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string strRightSideX = "unSrc[i1 + j] * srcCoefX + (unSrc[(i1 - dimY) + j] - 2 * unSrc[i1 + j] + unSrc[(i1 + dimY) + j]) * operatorLxxCoef";
			if (fn != null) strRightSideX += " + fnCoefX * fn[i1 + j]";
			string strProgram = strDefinesProgonkaX + strProgramHeader + String.Format(programSourceProgonkaX, strRightSideX);

			ProgramOCL program = createProgram(strProgram);
			kernels[0] = program.CreateKernel(functionName);

			kernels[0].SetMemoryArgument(0, unOCL0);
			kernels[0].SetMemoryArgument(1, unOCLm);
			kernels[0].SetMemoryArgument(2, alphaXOCL);
			kernels[0].SetValueArgument(3, srcCoefX);
			kernels[0].SetValueArgument(4, operatorLxxCoef);
			kernels[0].SetValueArgument(5, fnCoefX);
			if (fn != null) kernels[0].SetMemoryArgument(6, fn);
		}

		void createProgramProgonkaY(T srcCoefY, T operatorLyyCoef, T fnCoefY)
		{
			string functionName = "ProgonkaY";
			string args = "(global {0} *unSrc, global {0} *unDst, global {0} *alphaY, {0} srcCoefY, {0} operatorLyyCoef, {0} fnCoefY" + (fn != null ? ", global {0} *fn)" : ")");
			args = string.Format(args, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string strRightSideY = "unSrc[i + j] * srcCoefY + (unSrc[i + (j - 1)] - 2 * unSrc[i + j] + unSrc[i + (j + 1)]) * operatorLyyCoef";
			if (fn != null) strRightSideY += " + fnCoefY * fn[i + j]";
			string strProgram = strDefinesProgonkaY + strProgramHeader + String.Format(programSourceProgonkaY, strRightSideY);

			ProgramOCL program = createProgram(strProgram);
			kernels[1] = program.CreateKernel(functionName);

			kernels[1].SetMemoryArgument(0, unOCLm);
			kernels[1].SetMemoryArgument(1, unOCL1);
			kernels[1].SetMemoryArgument(2, alphaYOCL);
			kernels[1].SetValueArgument(3, srcCoefY);
			kernels[1].SetValueArgument(4, operatorLyyCoef);
			kernels[1].SetValueArgument(5, fnCoefY);
			if (fn != null) kernels[1].SetMemoryArgument(6, fn);
		}
	}
}

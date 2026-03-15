using Cloo;
using DD128Numeric;
using QD256Numeric;
using System;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class EvolutionalFactorizationProgramsOCL
	{
		static public KernelOCL createProgramProgonkaX<T>(int dimX, int dimY, bool withFn, ContextOCL ctx, DeviceOCL device)
		{
			string functionName = "ProgonkaX";
			string args = "(global {0} *src, global {0} *dst, global {0} *alphaX, {0} stepX2, {0} twoDivTau, {0} ratio" + (withFn ? ", global {0} *fn)" : ")");
			args = string.Format(args, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;

			string operatorLxx = "(src[(i1 - dimY) + j] - 2.0 * src[i1 + j] + src[(i1 + dimY) + j])";
			string operatorLyy = "(src[i1 + (j - 1)] - 2.0 * src[i1 + j] + src[i1 + (j + 1)])";
			string strRightSideX = string.Format("twoDivTau * ({0} + {1} * ratio + {2})", operatorLxx, operatorLyy, withFn ? "fn[i1 + j] * stepX2" : "0");
			strDefinesProgonkaX = string.Format(definesProgonkaX, dimX, dimY);
			string strProgram = strDefinesProgonkaX + strProgramHeader + String.Format(programSourceProgonkaX, strRightSideX);

			ProgramOCL program = createProgram<T>(strProgram, ctx, device);
			return program.CreateKernel(functionName);
		}

		static public KernelOCL createProgramProgonkaY<T>(int dimY, ContextOCL ctx, DeviceOCL device)
		{
			string functionName = "ProgonkaY";
			string args = "(global {0} *src, global {0} *dst, global {0} *srcX, global {0} *alphaY, {0} hy2, {0} twoDivTau, {0} tau)";
			args = string.Format(args, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;

			string strRightSideY = "src[i + j] * twoDivTau * hy2";
			strDefinesProgonkaY = string.Format(definesProgonkaY, dimY);
			string strProgram = strDefinesProgonkaY + strProgramHeader + String.Format(programSourceProgonkaY, strRightSideY);

			ProgramOCL program = createProgram<T>(strProgram, ctx, device);
			return program.CreateKernel(functionName);
		}

		static ProgramOCL createProgram<T>(string strProgram, ContextOCL ctx, DeviceOCL device)
		{
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;
			//string compileOptions = typeof(T) == typeof(QD256) ? "-cl-opt-disable" : null;
			string compileOptions = null;//doesn't cause InvalidCommandQueueException

			return UtilsCL.createProgram(strProgram, compileOptions, ctx, device);//"-cl-opt-disable"
		}

		static string strDefinesProgonkaX, strDefinesProgonkaY;
		static string definesProgonkaX =
@"
#define dimX			{0}
#define dimY			{1}
#define dimXminus2	(dimX - 2)//else will replaced with sub_HD(i!=dimX, 2)
";

		static string programSourceProgonkaX =
@"
{{
	int j = get_global_id(0);//indeces are 1-based, workgroup indeces are 1-based

	for (int i = 1,i1 = dimY; i < dimX - 1; i++,i1 += dimY) dst[i1 + j] = HP(alphaX[i] * ({0} + ((i != 1) ? dst[i1 - dimY + j] : Zero)));//src is used inside strRightSideX
	for (int i = dimXminus2,i1 = dimY * dimXminus2; i > 0; i--,i1 -= dimY) dst[i1 + j] = HP(dst[i1 + j] + alphaX[i] * ((i != dimXminus2) ? dst[i1 + dimY + j] : Zero));
}}";

		static string definesProgonkaY =
@"
#define dimY	{0}
#define dimYminus2	(dimY - 2)//else will replaced with sub_HD(i!=dimY, 2)
";

		static string programSourceProgonkaY =
@"
{{
	int i = get_global_id(0);//indeces are 1-based, workgroup indeces are 1-based

	i *= dimY;
	for (int j = 1; j < dimY - 1; j++) dst[i + j] = HP(alphaY[j] * ({0} + ((j != 1) ? dst[i + j - 1] : Zero)));//src is used inside strRightSideY
	for (int j = dimYminus2; j > 0; j--) dst[i + j] = HP(dst[i + j] + alphaY[j] * ((j != dimYminus2) ? dst[i + j + 1] : Zero));
	//post proccess progonka
	for (int j = 1; j <= dimYminus2; j++) dst[i + j] = HP(srcX[i + j] + tau * dst[i + j]);
}}";
	}
}

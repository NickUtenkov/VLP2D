using System;
using VLP2D.Common;

namespace VLP2D.Model
{
	static class SplittingProgramsCU
	{
		static public string createProgramProgonkaX<T>(string functionName, bool withFn)
		{
			string args0 = "({0} *src, {0} *dst, {0} *alphaX";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string strRightSideX = "src[i1 + j] * srcCoefX + (src[(i1 - dimY) + j] - 2 * src[i1 + j] + src[(i1 + dimY) + j]) * operatorLxxCoef";
			if (withFn) strRightSideX += " + fnCoefX * fn[i1 + j]";

			return strProgramHeader + String.Format(ProgonkaCU.programSourceProgonkaX, strRightSideX);
		}

		static public string createProgramProgonkaY<T>(string functionName, bool withFn)
		{
			string args0 = "({0} *src, {0} *dst, {0} *alphaY";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string strRightSideY = "src[i + j] * srcCoefY + (src[i + (j - 1)] - 2 * src[i + j] + src[i + (j + 1)]) * operatorLyyCoef";
			if (withFn) strRightSideY += " + fnCoefY * fn[i + j]";

			return strProgramHeader + String.Format(ProgonkaCU.programSourceProgonkaY, strRightSideY);
		}
	}
}

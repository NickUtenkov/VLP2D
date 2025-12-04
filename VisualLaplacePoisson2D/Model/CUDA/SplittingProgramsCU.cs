using System;
using VLP2D.Common;

namespace VLP2D.Model
{
	static class SplittingProgramsCU
	{
		static public string createProgramProgonkaX<T>(string functionName, bool withFn)
		{
			string args0 = "({0} *unSrc, {0} *unDst, {0} *alphaX";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string strRightSideX = "unSrc[i1 + j] * srcCoefX + (unSrc[(i1 - dimY) + j] - 2 * unSrc[i1 + j] + unSrc[(i1 + dimY) + j]) * operatorLxxCoef";
			if (withFn) strRightSideX += " + fnCoefX * fn[i1 + j]";

			return strProgramHeader + String.Format(ProgonkaCU.programSourceProgonkaX, strRightSideX);
		}

		static public string createProgramProgonkaY<T>(string functionName, bool withFn)
		{
			string args0 = "({0} *unSrc, {0} *unDst, {0} *alphaY";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string strRightSideY = "unSrc[i + j] * srcCoefY + (unSrc[i + (j - 1)] - 2 * unSrc[i + j] + unSrc[i + (j + 1)]) * operatorLyyCoef";
			if (withFn) strRightSideY += " + fnCoefY * fn[i + j]";

			return strProgramHeader + String.Format(ProgonkaCU.programSourceProgonkaY, strRightSideY);
		}
	}
}

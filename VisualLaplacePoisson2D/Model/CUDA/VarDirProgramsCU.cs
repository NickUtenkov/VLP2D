using System;
using VLP2D.Common;

namespace VLP2D.Model
{
	static class VarDirProgramsCU
	{
		static public string createProgramProgonkaX<T>(string functionName, bool withFn, bool equalSteps)
		{
			string kernelHeader = UtilsCU.kernelPrefix + functionName;
			string args0 = "({0} *unSrc, {0} *unDst, {0} *alphaX, {0} srcCoefX";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = kernelHeader + args;
			string unMult = "unSrc[i1 + j] * srcCoefX";
			string operatorLyy = "(unSrc[i1 + (j - 1)] - 2 * unSrc[i1 + j] + unSrc[i1 + (j + 1)])";
			string strRightSideX;

			string term1 = equalSteps ? operatorLyy : string.Format("({0} * {1})", operatorLyy, "stepX2DivY2");
			strRightSideX = string.Format("({0} + {1})", unMult, term1);
			if (withFn) strRightSideX = string.Format("({0} + {1})", strRightSideX, "fn[i1 + j] * stepX2");

			return strProgramHeader + String.Format(ProgonkaCU.programSourceProgonkaX, strRightSideX);
		}

		static public string createProgramProgonkaY<T>(string functionName, bool withFn, bool equalSteps)
		{
			string kernelHeader = UtilsCU.kernelPrefix + functionName;
			string args0 = "({0} *unSrc, {0} *unDst, {0} *alphaY, {0} srcCoefY";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = kernelHeader + args;
			string unMult = "(unSrc[i + j] * srcCoefY)";
			string operatorLxx = "(unSrc[(i - dimY) + j] - 2 * unSrc[i + j] + unSrc[(i + dimY) + j])";
			string strRightSideY;

			string term1 = equalSteps ? operatorLxx : string.Format("({0} * {1})", operatorLxx, "stepY2DivX2");
			strRightSideY = string.Format("({0} + {1})", unMult, term1);
			if (withFn) strRightSideY = string.Format("({0} + {1})", strRightSideY, "fn[i + j] * stepY2");

			return strProgramHeader + String.Format(ProgonkaCU.programSourceProgonkaY, strRightSideY);
		}
	}
}

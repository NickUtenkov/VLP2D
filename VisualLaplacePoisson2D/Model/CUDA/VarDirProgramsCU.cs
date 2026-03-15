using System;
using VLP2D.Common;

namespace VLP2D.Model
{
	static class VarDirProgramsCU
	{
		static public string createProgramProgonkaX<T>(string functionName, bool withFn)
		{
			string kernelHeader = UtilsCU.kernelPrefix + functionName;
			string args0 = "({0} *src, {0} *dst, {0} *alphaX, {0} coef";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = kernelHeader + args;

			string strRightSideX = string.Format("stepX2 * (src[i1 + j] * coef + (src[i1 + (j - 1)] - 2 * src[i1 + j] + src[i1 + (j + 1)]) * oneDivY2 + {0})", withFn ? "fn[i1 + j]" : "0");
			return strProgramHeader + String.Format(ProgonkaCU.programSourceProgonkaX, strRightSideX);
		}

		static public string createProgramProgonkaY<T>(string functionName, bool withFn)
		{
			string kernelHeader = UtilsCU.kernelPrefix + functionName;
			string args0 = "({0} *src, {0} *dst, {0} *alphaY, {0} coef";
			args0 += withFn ? ", {0} *fn)" : ")";
			string args = string.Format(args0, Utils.getTypeName<T>());
			string strProgramHeader = kernelHeader + args;

			string strRightSideY = string.Format("stepY2 * (src[i + j] * coef + (src[(i - dimY) + j] - 2 * src[i + j] + src[(i + dimY) + j]) * oneDivX2 + {0})", withFn ? "fn[i + j]" : "0");
			return strProgramHeader + String.Format(ProgonkaCU.programSourceProgonkaY, strRightSideY);
		}
	}
}

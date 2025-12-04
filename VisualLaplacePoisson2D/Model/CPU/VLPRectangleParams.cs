
using System;
using System.Collections.Generic;

namespace VLP2D.Model
{
	public class VLPRectangleParams
	{
		public int cXSegmentsOriginal, cYSegmentsOriginal;
		public int cXSegments, cYSegments;
		public double stepX, stepY;
		public Dictionary<string, int[]> maxCPUIters;
		public Dictionary<string, int[]> maxOCLIters;
		public Dictionary<string, int[]> maxCUDAIters;
		public Dictionary<string, double> dictEps;
		public string fKsi;
		public string funcLeft, funcRight, funcTop, funcBottom;
		public string funcBoundary, funcAnalytic;
		public string xMax, yMax;
		public String name { get; set; }//property because of binding

		public VLPRectangleParams(string xMax, string yMax, int cXSegments, int cYSegments, Dictionary<string, int[]> itersCPU, Dictionary<string, int[]> itersOCL, Dictionary<string, int[]> itersCUDA, Dictionary<string, double> dictEps, string fKsi, string funcLeft, string funcRight, string funcTop, string funcBottom, string funcBoundary, string funcAnalytic, string name)
		{
			this.xMax = xMax;
			this.yMax = yMax;
			cXSegmentsOriginal = cXSegments;
			cYSegmentsOriginal = cYSegments;
			this.cXSegments = cXSegments;
			this.cYSegments = cYSegments;

			maxCPUIters = itersCPU;
			maxOCLIters = itersOCL;
			maxCUDAIters = itersCUDA;
			this.dictEps = dictEps;
			this.fKsi = fKsi;
			this.funcLeft = funcLeft;
			this.funcRight = funcRight;
			this.funcTop = funcTop;
			this.funcBottom = funcBottom;
			this.funcBoundary = funcBoundary;
			this.funcAnalytic = funcAnalytic;
			this.name = name;
		}
	}
}

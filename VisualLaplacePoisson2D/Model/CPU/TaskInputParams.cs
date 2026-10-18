
using System;
using System.Collections.Generic;

namespace VLP2D.Model
{
	public class TaskInputParams
	{
		public int cXSegments, cYSegments;
		public Dictionary<string, int[]> maxCPUIters;
		public Dictionary<string, int[]> maxOCLIters;
		public Dictionary<string, int[]> maxCUDAIters;
		public Dictionary<string, double> dictEps;
		public string fKsi;
		public string funcLeft, funcRight, funcTop, funcBottom;
		public string funcBoundary, funcAnalytic;
		public string series1Function, series1Multiplier, series1IndexMin, series1IndexMax, series1Step;
		public string series2Function, series2Multiplier, series2IndexMin, series2IndexMax, series2Step;
		public string xMin, yMin;
		public string xMax, yMax;
		public String name { get; set; }//property because of binding

		public TaskInputParams(string xMin, string yMin, string xMax, string yMax, int cXSegments, int cYSegments, Dictionary<string, int[]> itersCPU,
			Dictionary<string, int[]> itersOCL, Dictionary<string, int[]> itersCUDA, Dictionary<string, double> dictEps, string fKsi,
			string funcLeft, string funcRight, string funcTop, string funcBottom, string funcBoundary, string funcAnalytic,
			string series1Function, string series1Multiplier, string series1IndexMin, string series1IndexMax, string series1Step,
			string series2Function, string series2Multiplier, string series2IndexMin, string series2IndexMax, string series2Step,
			string name)
		{
			this.xMin = xMin;
			this.yMin = yMin;
			this.xMax = xMax;
			this.yMax = yMax;
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

			this.series1Function = series1Function;
			this.series1Multiplier = series1Multiplier;
			this.series1IndexMin = series1IndexMin;
			this.series1IndexMax = series1IndexMax;
			this.series1Step = series1Step;

			this.series2Function = series2Function;
			this.series2Multiplier = series2Multiplier;
			this.series2IndexMin = series2IndexMin;
			this.series2IndexMax = series2IndexMax;
			this.series2Step = series2Step;

			this.name = name;
		}
	}
}

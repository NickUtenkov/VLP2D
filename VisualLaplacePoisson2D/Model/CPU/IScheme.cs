
using System;
using System.Windows.Media.Imaging;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	public enum IterationsKind
	{
		None,
		knownInAdvance,
		unknown
	}

	interface ISchemeInit<T>
	{
		(int, int) getArrayDimensions();
		void initTopBottomBorders(T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax);
		void initLeftRightBorders(T deltaX, T deltaY, Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax);
		void pointsMinMax(ref T valMin, ref T valMax);
		void initInitialIterationMean(T val);
		void initInitialIterationArithmeticMean();
		void initInitialIterationLinearInterpolation();
		void initInitialIterationWeightLinearInterpolation();
		BitmapSource createBitmap(MinMaxF minMax, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap);
		void calculateDifference(T[][] unDiff, T stpX, T stpY, Func<T, T, T> funcAnalitic, ref T fMin, ref T fMax, Func<bool> canceled, Action<double> reportProgress);

		IterationsKind iterationsKind();
		string getElapsedInfo();
	}

	interface ISchemeCalculate<T>
	{
		T doIteration(int iter);
		void initAfterBoundariesAndInitialIterationInited();
		void cancelIterations();
		int maxIterations();
		bool shouldReportProgress();
		void cleanup();
	}

	interface IScheme<T> : ISchemeInit<T>, ISchemeCalculate<T>
	{
	}
}

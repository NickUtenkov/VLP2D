using System;
using System.Numerics;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	public class Iterative2DScheme<T> where T : INumber<T>, IMinMaxValue<T>
	{
		protected T[,] un0;
		public virtual T[,] getArray2D() { return un0; }

		public (int, int) getArrayDimensions()
		{
			return (un0.GetUpperBound(0) + 1, un0.GetUpperBound(1) + 1);
		}

		public void initTopBottomBorders(T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			UtilsBorders.initTopBottomBorders<T>(getArray2D(), deltaX, deltaY, funcBottom, funcTop, funcBorder, ref valMin, ref valMax);
		}

		public void initLeftRightBorders(T deltaX, T deltaY, Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			UtilsBorders.initLeftRightBorders(getArray2D(), deltaX, deltaY, funcLeft, funcRight, funcBorder, ref valMin, ref valMax);
		}

		public void pointsMinMax(ref T valMin, ref T valMax)
		{
			T[,] ar = getArray2D();
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => ar[i, j]);
			UtilsDiff.pointsMinMax(adapter, ref valMin, ref valMax);
		}

		public BitmapSource createBitmap(MinMaxF minMax, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			T[,] ar = getArray2D();
			int dim1 = ar.GetUpperBound(0) + 1;
			int dim2 = ar.GetUpperBound(1) + 1;
			return fCreateBitmap(false, minMax, new Adapter2D<float>(dim1, dim2, (i, j) => float.CreateTruncating(ar[i, j])));
		}

		public void initInitialIterationMean(T val)
		{
			UtilsII.initInitialIterationMean<T>(getArray2D(), val);
		}

		public void initInitialIterationArithmeticMean()
		{
			UtilsII.initInitialIterationArithmeticMean(getArray2D());
		}

		public void initInitialIterationLinearInterpolation()
		{
			UtilsII.initInitialIterationLinearInterpolation(getArray2D());
		}

		public void initInitialIterationWeightLinearInterpolation()
		{
			UtilsII.initInitialIterationWeightLinearInterpolation(getArray2D());
		}

		public void calculateDifference(T[][] unDiff, T stpX, T stpY, Func<T, T, T> funcAnalitic, ref T fMin, ref T fMax, Func<bool> canceled, Action<double> reportProgress)
		{
			T[,] ar = getArray2D();
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => ar[i, j]);
			UtilsDiff.calculateDifference<T>(adapter, unDiff, stpX, stpY, funcAnalitic, ref fMin, ref fMax, canceled, reportProgress);
		}

		public string getElapsedInfo() { return null; }

		public virtual IterationsKind iterationsKind()
		{
			return IterationsKind.unknown;
		}
	}
}

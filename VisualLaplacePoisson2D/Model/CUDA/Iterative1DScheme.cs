using System;
using System.Numerics;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	public class Iterative1DScheme<T> where T : INumber<T>, IMinMaxValue<T>
	{
		protected int dimX, dimY;
		public virtual T[] getArray() { return null; }

		public (int, int) getArrayDimensions()
		{
			return (dimX, dimY);
		}

		public void initTopBottomBorders(T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			UtilsBorders.initTopBottomBorders(getArray(), dimX, dimY, deltaX, deltaY, funcBottom, funcTop, funcBorder, ref valMin, ref valMax);
		}

		public void initLeftRightBorders(T deltaX, T deltaY, Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			UtilsBorders.initLeftRightBorders(getArray(), dimX, dimY, deltaX, deltaY, funcLeft, funcRight, funcBorder, ref valMin, ref valMax);
		}

		public void pointsMinMax(ref T valMin, ref T valMax)
		{
			T[] ar = getArray();
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => ar[i * dims.Item2 + j]);
			UtilsDiff.pointsMinMax(adapter, ref valMin, ref valMax);
		}

		public BitmapSource createBitmap(MinMaxF minMax, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			T[] ar = getArray();
			return ar != null ? fCreateBitmap(false, minMax, new Adapter2D<float>(dimX, dimY, (i, j) => float.CreateTruncating(ar[i * dimY + j]))) : null;
		}

		public void initInitialIterationMean(T val)
		{
			UtilsII.initInitialIterationMean<T>(getArray(), dimX, dimY, val);
		}

		public void initInitialIterationArithmeticMean()
		{
			UtilsII.initInitialIterationArithmeticMean(getArray(), dimX, dimY);
		}

		public void initInitialIterationLinearInterpolation()
		{
			UtilsII.initInitialIterationLinearInterpolation(getArray(), dimX, dimY);
		}

		public void initInitialIterationWeightLinearInterpolation()
		{
			UtilsII.initInitialIterationWeightLinearInterpolation(getArray(), dimX, dimY);
		}

		public void calculateDifference(T[][] unDiff, T stpX, T stpY, Func<T, T, T> funcAnalitic, ref T fMin, ref T fMax, Func<bool> canceled, Action<double> reportProgress)
		{
			T[] ar = getArray();
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => ar[i * dims.Item2 + j]);
			UtilsDiff.calculateDifference(adapter, unDiff, stpX, stpY, funcAnalitic, ref fMin, ref fMax, canceled, reportProgress);
		}

		virtual public string getElapsedInfo() { return null; }

		public virtual IterationsKind iterationsKind()
		{
			return IterationsKind.unknown;
		}
	}
}

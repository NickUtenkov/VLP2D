using System;
using System.Numerics;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class DirectJagged2Scheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		protected int N1, N2;//Upper Bounds
		protected T[][] un;
		readonly bool isLaplace;
		protected MinMaxF minMax;

		public DirectJagged2Scheme(int dim1, int dim2, bool isLaplace)
		{
			N1 = dim1 - 1;
			N2 = dim2 - 1;
			un = new T[dim1][];
			for (int i = 0; i < dim1; i++) un[i] = new T[dim2];
			this.isLaplace = isLaplace;
			minMax = null;
		}

		public (int, int) getArrayDimensions()
		{
			return (N1 + 1, N2 + 1);//== (dim1, dim2)
		}

		public void initTopBottomBorders(T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			UtilsBorders.initTopBottomBorders(un, deltaX, deltaY, funcBottom, funcTop, funcBorder, ref valMin, ref valMax);
			updateMinMax(valMin, valMax);
		}

		public void initLeftRightBorders(T deltaX, T deltaY, Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			UtilsBorders.initLeftRightBorders(un, deltaX, deltaY, funcLeft, funcRight, funcBorder, ref valMin, ref valMax);
			updateMinMax(valMin, valMax);
		}

		public void pointsMinMax(ref T valMin, ref T valMax)
		{
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => un[i][j]);
			UtilsDiff.pointsMinMax(adapter, ref valMin, ref valMax);
		}

		public BitmapSource createBitmap(MinMaxF minMax, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			return fCreateBitmap(true, minMax, new Adapter2D<float>(N1 + 1, N2 + 1, (i, j) => float.CreateTruncating(un[i][j])));
		}

		public void calculateDifference(T[][] unDiff, T stpX, T stpY, Func<T, T, T> funcAnalitic, ref T fMin, ref T fMax, Func<bool> canceled, Action<double> reportProgress)
		{
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => un[i][j]);
			UtilsDiff.calculateDifference(adapter, unDiff, stpX, stpY, funcAnalitic, ref fMin, ref fMax, canceled, reportProgress);
		}

		public void initInitialIterationMean(T val) { }
		public void initInitialIterationArithmeticMean() { }
		public void initInitialIterationLinearInterpolation() { }
		public void initInitialIterationWeightLinearInterpolation() { }

		public virtual string getElapsedInfo() { return null; }

		public virtual void cleanup()
		{
			un = null;
		}

		void updateMinMax(T valMin, T valMax)
		{
			if (isLaplace)
			{
				if (minMax == null) minMax = new MinMaxF();
				minMax.min = float.CreateTruncating(valMin);
				minMax.max = float.CreateTruncating(valMax);
			}
		}

		public IterationsKind iterationsKind()
		{
			return IterationsKind.None;
		}
	}
}

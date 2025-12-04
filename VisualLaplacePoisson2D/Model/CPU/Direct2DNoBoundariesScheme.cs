using System;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	public class Direct2DNoBoundariesScheme<T> where T : INumber<T>, IMinMaxValue<T>
	{
		protected T[,] un;
		protected int dim1, dim2;
		protected T stepX2, stepY2;
		protected ParallelOptions optionsParallel;

		public Direct2DNoBoundariesScheme(int rows, int cols, T stepX, T stepY, ParallelOptions optionsParallel)
		{
			this.optionsParallel = optionsParallel;

			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;

			dim1 = rows;
			dim2 = cols;
			un = new T[dim1, dim2];
		}

		~Direct2DNoBoundariesScheme()
		{
			un = null;
		}

		public (int, int) getArrayDimensions()
		{
			return (dim1 + 1, dim2 + 1);
		}

		public void initTopBottomBorders(T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			T yMax = deltaY * T.CreateTruncating(dim2 + 1);
			int cLoop = Math.Min(dim1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incX = deltaX * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T x = deltaX * T.CreateTruncating(core + 1);
				for (int i = 0 + core; i < dim1; i += cLoop)
				{
					T val0 = ((funcBottom != null) ? funcBottom(x) : funcBorder(x, T.Zero));
					un[i, 0] += (val0 / stepY2);//[SNR] p.190, (19)

					T val1 = ((funcTop != null) ? funcTop(x) : funcBorder(x, yMax));
					un[i, dim2 - 1] += (val1 / stepY2);//[SNR] p.190, (19)

					UtilsDiff.updateMinMax(val0, ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(val1, ref fMin[core], ref fMax[core]);

					x += incX;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
		}

		public void initLeftRightBorders(T deltaX, T deltaY, Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			T xMax = deltaX * T.CreateTruncating(dim1 + 1);
			int cLoop = Math.Min(dim2, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incY = deltaY * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T y = deltaY * T.CreateTruncating(core + 1);
				for (int j = 0 + core; j < dim2; j += cLoop)
				{
					T val0 = (funcLeft != null) ? funcLeft(y) : funcBorder(T.Zero, y);
					un[0, j] += (val0 / stepX2);//[SNR] p.190, (19)

					T val1 = (funcRight != null) ? funcRight(y) : funcBorder(xMax, y);
					un[dim1 - 1, j] += (val1 / stepX2);//[SNR] p.190, (19)

					UtilsDiff.updateMinMax(val0, ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(val1, ref fMin[core], ref fMax[core]);

					y += incY;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
		}

		public void pointsMinMax(ref T valMin, ref T valMax)
		{
			T valMin1 = T.MaxValue, valMax1 = T.MinValue;
			iterate((i, j) => UtilsDiff.updateMinMax(un[i, j], ref valMin1, ref valMax1));
			valMin = valMin1;
			valMax = valMax1;
		}

		public BitmapSource createBitmap(MinMaxF minMax, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			return fCreateBitmap(true, minMax, new Adapter2D<float>(dim1, dim2, (i, j) => float.CreateTruncating(un[i, j])));
		}

		public void calculateDifference(T[][] unDiff, T stpX, T stpY, Func<T, T, T> funcAnalitic, ref T valMin, ref T valMax, Func<bool> canceled, Action<double> reportProgress)
		{
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => un[i - 1, j - 1]);
			UtilsDiff.calculateDifference(adapter, unDiff, stpX, stpY, funcAnalitic, ref valMin, ref valMax, canceled, reportProgress);
		}

		public void iterate(Action<int, int> func)
		{
			Parallel.For(0, dim1, optionsParallel, i =>
			{
				for (int j = 0; j < dim2; j++)
				{
					func(i, j);
				}
			});
		}

		public void initInitialIterationMean(T val) { }
		public void initInitialIterationArithmeticMean() { }
		public void initInitialIterationLinearInterpolation() { }
		public void initInitialIterationWeightLinearInterpolation() { }
		virtual public string getElapsedInfo() { return null; }

		public IterationsKind iterationsKind()
		{
			return IterationsKind.None;
		}
	}
}

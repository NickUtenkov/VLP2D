using System;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	public class Direct1DNoBoundariesScheme<T> where T : INumber<T>, IMinMaxValue<T>
	{
		protected T[] un;
		protected int dim1, dim2;
		protected T stepX2, stepY2;
		protected ParallelOptions optionsParallel;

		public Direct1DNoBoundariesScheme(int rows, int cols, T stepX, T stepY, ParallelOptions optionsParallel)
		{
			this.optionsParallel = optionsParallel;

			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;

			dim1 = rows;
			dim2 = cols;
			un = new T[dim1 * dim2];
		}

		public (int, int) getArrayDimensions()
		{
			return (dim1 + 1, dim2 + 1);//assumes calculateDifference indices begins from 1
		}

		public void initTopBottomBorders(T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			//В этой схеме рабочий массив un не содержит границ. Граничные значения хранить негде(или надо использовать дополнительные массивы).
			//Поэтому вычисленные значения на границах(поделенные на квадраты шагов) сразу добавляются в приграничные узлы.
			//In this scheme, the working array un does not contain boundaries. There is no place to store boundary values ​​(or additional arrays must be used).
			//Therefore, the calculated boundary values ​​(divided by the squared steps) are immediately added to the near boundary nodes.
			T yMax = deltaY * T.CreateTruncating(dim2 + 1);
			int cLoop = Math.Min(dim1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incX = deltaX * T.CreateTruncating(cLoop);
			int idxB = 0;
			int idxT = dim2 - 1;
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T x = deltaX * T.CreateTruncating(core + 1);
				for (int i = 0 + core; i < dim1; i += cLoop)
				{
					T valB = ((funcBottom != null) ? funcBottom(x) : funcBorder(x, T.Zero));
					un[i * dim2 + idxB] += valB / stepY2;//[SNR] p.190, (19)

					T valT = ((funcTop != null) ? funcTop(x) : funcBorder(x, yMax));
					un[i * dim2 + idxT] += valT / stepY2;//[SNR] p.190, (19)

					UtilsDiff.updateMinMax(valB, ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(valT, ref fMin[core], ref fMax[core]);

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
			//В этой схеме рабочий массив un не содержит границ. Граничные значения хранить негде(или надо использовать дополнительные массивы).
			//Поэтому вычисленные значения на границах(поделенные на квадраты шагов) сразу добавляются в приграничные узлы.
			//In this scheme, the working array un does not contain boundaries. There is no place to store boundary values ​​(or additional arrays must be used).
			//Therefore, the calculated boundary values ​​(divided by the squared steps) are immediately added to the near boundary nodes.
			T xMax = deltaX * T.CreateTruncating(dim1 + 1);
			int cLoop = Math.Min(dim2, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incY = deltaY * T.CreateTruncating(cLoop);
			int idxL = 0 * dim2;
			int idxR = (dim1 - 1) * dim2;
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T y = deltaY * T.CreateTruncating(core + 1);
				for (int j = 0 + core; j < dim2; j += cLoop)
				{
					T valL = (funcLeft != null) ? funcLeft(y) : funcBorder(T.Zero, y);
					un[idxL + j] += valL / stepX2;//[SNR] p.190, (19)

					T valR = (funcRight != null) ? funcRight(y) : funcBorder(xMax, y);
					un[idxR + j] += valR / stepX2;//[SNR] p.190, (19)

					UtilsDiff.updateMinMax(valL, ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(valR, ref fMin[core], ref fMax[core]);

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
			iterate((i, j) => UtilsDiff.updateMinMax(un[i * dim2 + j], ref valMin1, ref valMax1));
			valMin = valMin1;
			valMax = valMax1;
		}

		public BitmapSource createBitmap(MinMaxF minMax, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			return fCreateBitmap(true, minMax, new Adapter2D<float>(dim1, dim2, (i, j) => float.CreateTruncating(un[i * dim2 + j])));
		}

		public void calculateDifference(T[][] unDiff, T stpX, T stpY, Func<T, T, T> funcAnalitic, ref T valMin, ref T valMax, Func<bool> canceled, Action<double> reportProgress)
		{
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => un[(i - 1) * dim2 + (j - 1)]);//assumes calculateDifference indices begins from 1
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

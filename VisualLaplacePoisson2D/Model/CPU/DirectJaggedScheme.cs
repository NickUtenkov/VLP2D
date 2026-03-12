using System;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class DirectJaggedScheme<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		protected T xMin, yMin, stepX, stepY;
		protected readonly int Nx, Ny;
		protected T[][] un;
		protected T[] bndL, bndR, bndT, bndB;//left, right, top, bottom boundaries
		protected T unLB, unLT, unRB, unRT;
		protected readonly Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;

		public DirectJaggedScheme(int cXSegments, int cYSegments, T xMin, T yMin, T stepX, T stepY, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			this.xMin = xMin;
			this.yMin = yMin;
			this.stepX = stepX;
			this.stepY = stepY;

			Nx = cXSegments;
			Ny = cYSegments;
			this.fCreateBitmap = fCreateBitmap;

			bndB = new T[Nx - 1];
			bndT = new T[Nx - 1];

			bndL = new T[Ny - 1];
			bndR = new T[Ny - 1];
		}

		public (int, int) getArrayDimensions()
		{
			return (Nx + 1, Ny + 1);
		}

		public void initTopBottomBorders(Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			T yMax = yMin + stepY * T.CreateTruncating(Ny);
			int cLoop = Math.Min(Nx - 1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incX = stepX * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T x = xMin + stepX * T.CreateTruncating(core + 1);//not use corner value
				for (int i = 0 + core; i <= Nx - 2; i += cLoop)
				{
					bndB[i] = (funcBottom != null) ? funcBottom(x) : funcBorder(x, yMin);
					bndT[i] = (funcTop != null) ? funcTop(x) : funcBorder(x, yMax);

					UtilsDiff.updateMinMax(bndB[i], ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(bndT[i], ref fMin[core], ref fMax[core]);

					x += incX;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
			unLB = (funcBottom != null) ? funcBottom(xMin) : funcBorder(xMin, yMin);//or funcLeft(yMin)
			unLT = (funcTop != null) ? funcTop(xMin) : funcBorder(xMin, yMax);//or funcLeft(yMax)
			T xMax = xMin + stepX * T.CreateTruncating(Nx);
			unRB = (funcBottom != null) ? funcBottom(yMax) : funcBorder(xMax, yMin);
			unRT = (funcTop != null) ? funcTop(yMax) : funcBorder(xMax, yMax);
		}

		public void initLeftRightBorders(Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax)
		{
			T xMax = xMin + stepX * T.CreateTruncating(Nx);
			int cLoop = Math.Min(Ny - 1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T incY = stepY * T.CreateTruncating(cLoop);
			Lock locker = new();
			T min = valMin;
			T max = valMax;
			Parallel.For(0, cLoop, GridIterator.optionsParallel, () => new MyRange<T>(min, max), (core, state, range) =>
			{
				T y = yMin + stepY * T.CreateTruncating(core + 1);//not use corner value
				for (int j = 0 + core; j <= Ny - 2; j += cLoop)
				{
					bndL[j] = (funcLeft != null) ? funcLeft(y) : funcBorder(xMin, y);
					bndR[j] = (funcRight != null) ? funcRight(y) : funcBorder(xMax, y);

					UtilsDiff.updateMinMax(bndL[j], ref range.min, ref range.max);
					UtilsDiff.updateMinMax(bndR[j], ref range.min, ref range.max);

					y += incY;
				}
				return range;
			},
			range =>
			{
				lock (locker)
				{
					UtilsDiff.updateMinMax(range.min, ref min, ref max);
					UtilsDiff.updateMinMax(range.max, ref min, ref max);
				}
			});
			valMin = min;
			valMax = max;
		}

		public void pointsMinMax(ref T valMin, ref T valMax)
		{
			T valMin1 = T.MaxValue, valMax1 = T.MinValue;
			GridIterator.iterate(0, un.GetUpperBound(0) + 1, 0, un[0].GetUpperBound(0) + 1, (i, j) => UtilsDiff.updateMinMax(un[i][j], ref valMin1, ref valMax1));
			valMin = (T)valMin1;
			valMax = (T)valMax1;
		}

		public BitmapSource createBitmap(MinMaxF minMax, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			return fCreateBitmap(true, minMax, new Adapter2D<float>(Nx + 1, Ny + 1, (i, j) => float.CreateTruncating(un[i][j])));
		}

		public void useAction(Action action)
		{
			action();
		}

		public void calculateDifference(T[][] unDiff, Func<T, T, T> funcAnalitic, ref T fMin, ref T fMax, Func<bool> canceled, Action<double> reportProgress)
		{
			(int, int) dims = getArrayDimensions();
			Adapter2D<T> adapter = new Adapter2D<T>(dims.Item1, dims.Item2, (i, j) => un[i][j]);
			UtilsDiff.calculateDifference(adapter, unDiff, xMin, yMin, stepX, stepY, funcAnalitic, ref fMin, ref fMax, canceled, reportProgress);
			if (Nx < 11) UtilsPrint.printJaggedArray(unDiff, "{0,11:E3}", "Diff");
		}

		public void initInitialIterationValue(T val) { }
		public void initInitialIterationArithmeticMean() { }
		public void initInitialIterationLinearInterpolation() { }
		public void initInitialIterationWeightLinearInterpolation() { }
		public virtual string getElapsedInfo() { return null; }

		public virtual void cleanup()
		{
			un = null;
			bndL = null;
			bndR = null;
			bndT = null;
			bndB = null;
		}

		public IterationsKind iterationsKind()
		{
			return IterationsKind.None;
		}
	}
}

using System;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class Direct1DSeparateBoundariesScheme<T> where T : struct, INumber<T>, IMinMaxValue<T>
	{
		protected T xMin, yMin, stepX, stepY;
		protected readonly int Nx, Ny;
		protected T[] un = null;
		protected T[] bndL, bndR, bndT, bndB;//left, right, top, bottom boundaries
		protected T unLB, unLT, unRB, unRT;
		protected readonly Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;

		public Direct1DSeparateBoundariesScheme(int cXSegments, int cYSegments, T xMin, T yMin, T stepX, T stepY, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
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
				T x = xMin + stepX * T.CreateTruncating(core + 1);
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
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incY = stepY * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T y = yMin + stepY * T.CreateTruncating(core + 1);
				for (int j = 0 + core; j <= Ny - 2; j += cLoop)
				{
					bndL[j] = (funcLeft != null) ? funcLeft(y) : funcBorder(xMin, y);
					bndR[j] = (funcRight != null) ? funcRight(y) : funcBorder(xMax, y);

					UtilsDiff.updateMinMax(bndL[j], ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(bndR[j], ref fMin[core], ref fMax[core]);

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
			int cLoop = Math.Min(Nx - 1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, T.MaxValue);
			Array.Fill(fMax, T.MinValue);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				for (int i = 0 + core; i <= Nx; i += cLoop)
				{
					for (int j = 0; j <= Ny; j++)
					{
						UtilsDiff.updateMinMax(un[i * (Ny + 1) + j], ref fMin[core], ref fMax[core]);
					}
				}
			});
			valMin = T.MaxValue;
			valMax = T.MinValue;
			for (int i = 0; i < cLoop; i++)
			{
				valMin = T.Min(fMin[i], valMin);
				valMax = T.Max(fMax[i], valMax);
			}
		}

		public BitmapSource createBitmap(MinMaxF minMax, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			if (un == null) return null;
			return fCreateBitmap(true, minMax, new Adapter2D<float>(Nx + 1, Ny + 1, (i, j) => float.CreateTruncating(un[i * (Ny + 1) + j])));
		}

		public void useAction(Action action)
		{
			action();
		}

		public void calculateDifference(T[][] unDiff, Func<T, T, T> funcAnalitic, ref T valMin, ref T valMax, Func<bool> canceled, Action<double> reportProgress)
		{
			if (un == null) return;
			Adapter2D<T> adapter = new Adapter2D<T>(Nx + 1, Ny + 1, (i, j) => un[i * (Ny + 1) + j]);
			UtilsDiff.calculateDifference(adapter, unDiff, xMin, yMin, stepX, stepY, funcAnalitic, ref valMin, ref valMax, canceled, reportProgress);
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

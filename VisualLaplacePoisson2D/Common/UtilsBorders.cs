using System;
using System.Numerics;
using System.Threading.Tasks;

namespace VLP2D.Common
{
	internal class UtilsBorders
	{
		public static void copyTopBottomValues<T>(T[,] unSrc, T[,] unDst)
		{
			int upperX = unSrc.GetUpperBound(0);
			int upperY = unSrc.GetUpperBound(1);
			for (int i = 0; i <= upperX; i++)
			{
				unDst[i, 0] = unSrc[i, 0];//copy boundary values
				unDst[i, upperY] = unSrc[i, upperY];//copy boundary values
			}
		}

		public static void copyTopBottomValues(double[,] unSrc, double[,] unDst)
		{
			int upperX = unSrc.GetUpperBound(0);
			int upperY = unSrc.GetUpperBound(1);
			for (int i = 0; i <= upperX; i++)
			{
				unDst[i, 0] = unSrc[i, 0];//copy boundary values
				unDst[i, upperY] = unSrc[i, upperY];//copy boundary values
			}
		}

		public static void copyLeftRightValues<T>(T[,] unSrc, T[,] unDst) where T : INumber<T>
		{
			int upperX = unSrc.GetUpperBound(0);
			int upperY = unSrc.GetUpperBound(1);
			for (int j = 0; j <= upperY; j++)
			{
				unDst[0, j] = unSrc[0, j];//copy boundary values
				unDst[upperX, j] = unSrc[upperX, j];//copy boundary values
			}
		}

		public static void copyLeftRightValues(double[,] unSrc, double[,] unDst)
		{
			int upperX = unSrc.GetUpperBound(0);
			int upperY = unSrc.GetUpperBound(1);
			for (int j = 0; j <= upperY; j++)
			{
				unDst[0, j] = unSrc[0, j];//copy boundary values
				unDst[upperX, j] = unSrc[upperX, j];//copy boundary values
			}
		}

		public static void initTopBottomBorders<T>(T[,] unSrc, T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax) where T : INumber<T>, IMinMaxValue<T>
		{
			int dim1 = unSrc.GetUpperBound(0) + 1;
			int upperY = unSrc.GetUpperBound(1);
			T yMax = deltaY * T.CreateTruncating(upperY);
			int cLoop = Math.Min(dim1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incX = deltaX * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T x = deltaX * T.CreateTruncating(core);
				for (int i = 0 + core; i < dim1; i += cLoop)
				{
					unSrc[i, 0] = (funcBottom != null) ? funcBottom(x) : funcBorder(x, T.Zero);
					unSrc[i, upperY] = (funcTop != null) ? funcTop(x) : funcBorder(x, yMax);

					UtilsDiff.updateMinMax(unSrc[i, 0], ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(unSrc[i, upperY], ref fMin[core], ref fMax[core]);

					x += incX;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				if (fMin[i] != T.MaxValue) UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				if (fMax[i] != T.MinValue) UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
		}

		public static void initTopBottomBorders<T>(T[][] unSrc, T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax) where T : INumber<T>, IMinMaxValue<T>
		{
			int dim1 = unSrc.GetUpperBound(0) + 1;
			int upperY = unSrc[0].GetUpperBound(0);
			T yMax = deltaY * T.CreateTruncating(upperY);
			int cLoop = Math.Min(dim1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incX = deltaX * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T x = deltaX * T.CreateTruncating(core);
				for (int i = 0 + core; i < dim1; i += cLoop)
				{
					unSrc[i][0] = (funcBottom != null) ? funcBottom(x) : funcBorder(x, T.Zero);
					unSrc[i][upperY] = (funcTop != null) ? funcTop(x) : funcBorder(x, yMax);

					UtilsDiff.updateMinMax(unSrc[i][0], ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(unSrc[i][upperY], ref fMin[core], ref fMax[core]);

					x += incX;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				if (fMin[i] != T.MaxValue) UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				if (fMax[i] != T.MinValue) UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
		}

		public static void initTopBottomBorders<T>(T[] unSrc, int dim1, int dim2, T deltaX, T deltaY, Func<T, T> funcBottom, Func<T, T> funcTop, Func<T, T, T> funcBorder, ref T valMin, ref T valMax) where T : INumber<T>, IMinMaxValue<T>
		{
			if (unSrc == null) return;
			int upperY = dim2 - 1;
			T yMax = deltaY * T.CreateTruncating(upperY);
			int cLoop = Math.Min(dim1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incX = deltaX * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T x = deltaX * T.CreateTruncating(core);
				for (int i = 0 + core; i < dim1; i += cLoop)
				{
					unSrc[i * dim2 + 0] = (funcBottom != null) ? funcBottom(x) : funcBorder(x, T.Zero);
					unSrc[i * dim2 + upperY] = (funcTop != null) ? funcTop(x) : funcBorder(x, yMax);

					UtilsDiff.updateMinMax(unSrc[i * dim2 + 0], ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(unSrc[i * dim2 + upperY], ref fMin[core], ref fMax[core]);

					x += incX;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
		}

		public static void initLeftRightBorders<T>(T[,] unSrc, T deltaX, T deltaY, Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax) where T : INumber<T>, IMinMaxValue<T>
		{
			if (unSrc == null) return;
			int upperX = unSrc.GetUpperBound(0);
			int dim2 = unSrc.GetUpperBound(1) + 1;
			T xMax = deltaX * T.CreateTruncating(upperX);
			int cLoop = Math.Min(dim2, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incY = deltaY * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T y = deltaY * T.CreateTruncating(core);
				for (int j = 0 + core; j < dim2; j += cLoop)
				{
					unSrc[0, j] = (funcLeft != null) ? funcLeft(y) : funcBorder(T.Zero, y);
					unSrc[upperX, j] = (funcRight != null) ? funcRight(y) : funcBorder(xMax, y);

					UtilsDiff.updateMinMax(unSrc[0, j], ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(unSrc[upperX, j], ref fMin[core], ref fMax[core]);

					y += incY;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				if (fMin[i] != T.MaxValue) UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				if (fMax[i] != T.MinValue) UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
		}

		public static void initLeftRightBorders<T>(T[][] unSrc, T deltaX, T deltaY, Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax) where T : INumber<T>, IMinMaxValue<T>
		{
			if (unSrc == null) return;
			int upperX = unSrc.GetUpperBound(0);
			int dim2 = unSrc[0].GetUpperBound(0) + 1;
			T xMax = deltaX * T.CreateTruncating(upperX);
			int cLoop = Math.Min(dim2, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incY = deltaY * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T y = deltaY * T.CreateTruncating(core);
				for (int j = 0 + core; j < dim2; j += cLoop)
				{
					unSrc[0][j] = (funcLeft != null) ? funcLeft(y) : funcBorder(T.Zero, y);
					unSrc[upperX][j] = (funcRight != null) ? funcRight(y) : funcBorder(xMax, y);

					UtilsDiff.updateMinMax(unSrc[0][j], ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(unSrc[upperX][j], ref fMin[core], ref fMax[core]);

					y += incY;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				if (fMin[i] != T.MaxValue) UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				if (fMax[i] != T.MinValue) UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
		}

		public static void initLeftRightBorders<T>(T[] unSrc, int dim1, int dim2, T deltaX, T deltaY, Func<T, T> funcLeft, Func<T, T> funcRight, Func<T, T, T> funcBorder, ref T valMin, ref T valMax) where T : INumber<T>, IMinMaxValue<T>
		{
			if (unSrc == null) return;
			int upperX = dim1 - 1;
			T xMax = deltaX * T.CreateTruncating(upperX);
			int cLoop = Math.Min(dim2, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, valMin);
			Array.Fill(fMax, valMax);
			T incY = deltaY * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T y = deltaY * T.CreateTruncating(core);
				for (int j = 0 + core; j < dim2; j += cLoop)
				{
					unSrc[0 * dim2 + j] = (funcLeft != null) ? funcLeft(y) : funcBorder(T.Zero, y);
					unSrc[upperX * dim2 + j] = (funcRight != null) ? funcRight(y) : funcBorder(xMax, y);

					UtilsDiff.updateMinMax(unSrc[0 * dim2 + j], ref fMin[core], ref fMax[core]);
					UtilsDiff.updateMinMax(unSrc[upperX * dim2 + j], ref fMin[core], ref fMax[core]);

					y += incY;
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				UtilsDiff.updateMinMax(fMin[i], ref valMin, ref valMax);
				UtilsDiff.updateMinMax(fMax[i], ref valMin, ref valMax);
			}
		}
	}
}

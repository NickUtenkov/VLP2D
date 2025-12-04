using System;
using System.Numerics;

namespace VLP2D.Common
{
	internal class UtilsII
	{
		public static void initInitialIterationMean<T>(T[,] unDst, T val)
		{
			int upper1 = unDst.GetUpperBound(0);
			int upper2 = unDst.GetUpperBound(1);
			GridIterator.iterate(upper1, upper2, (i, j) => unDst[i, j] = val);
		}

		public static void initInitialIterationMean<T>(T[] unDst, int dim1, int dim2, T val)
		{
			if (unDst == null) return;
			GridIterator.iterate(dim1 - 1, dim2 - 1, (i, j) => unDst[i * dim2 + j] = val);
		}

		public static void initInitialIterationArithmeticMean<T>(T[,] unDst) where T : INumberBase<T>//IAdditionOperators<T, T, T>, IDivisionOperators<T, double, T>
		{
			int upperX = unDst.GetUpperBound(0);
			int upperY = unDst.GetUpperBound(1);
			GridIterator.iterate(upperX, upperY, (i, j) => unDst[i, j] = (unDst[0, j] + unDst[upperX, j] + unDst[i, 0] + unDst[i, upperY]) / T.CreateTruncating(4));
		}

		public static void initInitialIterationArithmeticMean<T>(T[] unDst, int dim1, int dim2) where T : INumberBase<T>//IAdditionOperators<T, T, T>, IDivisionOperators<T, double, T>
		{
			int upperX = dim1 - 1;
			int upperY = dim2 - 1;
			GridIterator.iterate(dim1 - 1, dim2 - 1, (i, j) => unDst[i * dim2 + j] = (unDst[0 * dim2 + j] + unDst[upperX * dim2 + j] + unDst[i * dim2 + 0] + unDst[i * dim2 + upperY]) / T.CreateTruncating(4));
		}

		static T linearInterpolationX<T>(T[,] unDst, int i, int j) where T : INumberBase<T>//, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>
		{
			int upperX = unDst.GetUpperBound(0);
			T sigma = T.CreateTruncating(((float)i) / upperX);
			return unDst[0, j] * (T.One - sigma) + unDst[upperX, j] * sigma;
		}

		static T linearInterpolationY<T>(T[,] unDst, int i, int j) where T : INumberBase<T>//, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>
		{
			int upperY = unDst.GetUpperBound(1);
			T sigma = T.CreateTruncating(((float)j) / upperY);
			return unDst[i, 0] * (T.One - sigma) + unDst[i, upperY] * sigma;
		}

		public static void initInitialIterationLinearInterpolation<T>(T[,] unDst) where T : INumberBase<T>//, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>, IDivisionOperators<T, double, T>
		{
			int upper1 = unDst.GetUpperBound(0);
			int upper2 = unDst.GetUpperBound(1);
			GridIterator.iterate(upper1, upper2, (i, j) => unDst[i, j] = (linearInterpolationX(unDst, i, j) + linearInterpolationY(unDst, i, j)) / T.CreateTruncating(2));
		}

		static T linearInterpolationX<T>(T[] unDst, int i, int j, int dim1, int dim2) where T : INumberBase<T>//, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>
		{
			int upperX = dim1 - 1;
			T sigma = T.CreateTruncating(((float)i) / upperX);
			return unDst[0 * dim2 + j] * (T.One - sigma) + unDst[upperX * dim2 + j] * sigma;
		}

		static T linearInterpolationY<T>(T[] unDst, int i, int j, int dim1, int dim2) where T : INumberBase<T>//, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>
		{
			int upperY = dim2 - 1;
			T sigma = T.CreateTruncating(((float)j) / upperY);
			return unDst[i * dim2 + 0] * (T.One - sigma) + unDst[i * dim2 + upperY] * sigma;
		}

		public static void initInitialIterationLinearInterpolation<T>(T[] unDst, int dim1, int dim2) where T : INumberBase<T>//, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>, IDivisionOperators<T, double, T>
		{
			GridIterator.iterate(dim1 - 1, dim2 - 1, (i, j) => unDst[i * dim2 + j] = (linearInterpolationX(unDst, i, j, dim1, dim2) + linearInterpolationY(unDst, i, j, dim1, dim2)) / T.CreateTruncating(2));
		}

		public static void initInitialIterationWeightLinearInterpolation<T>(T[,] unDst) where T : INumberBase<T>//, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>, IDivisionOperators<T, double, T>
		{
			int cXSegments = unDst.GetUpperBound(0);
			int cYSegments = unDst.GetUpperBound(1);
			Func<int, int, T> func = (i, j) =>
			{
				int lngX = cXSegments > i + i ? i : cXSegments - i;//ie. i < cXSegments/2
				int lngY = cYSegments > j + j ? j : cYSegments - j;
				T weight = T.CreateTruncating((float)lngX / (lngX + lngY));
				return linearInterpolationX(unDst, i, j) * (T.One - weight) + linearInterpolationY(unDst, i, j) * weight;
			};
			int upper1 = unDst.GetUpperBound(0);
			int upper2 = unDst.GetUpperBound(1);
			GridIterator.iterate(upper1, upper2, (i, j) => unDst[i, j] = func(i, j));
		}

		public static void initInitialIterationWeightLinearInterpolation<T>(T[] unDst, int dim1, int dim2) where T : INumberBase<T>//, ISubtractionOperators<T, double, T>, IMultiplyOperators<T, double, T>, IDivisionOperators<T, double, T>
		{
			int cXSegments = dim1 - 1;
			int cYSegments = dim2 - 1;
			Func<int, int, T> func = (i, j) =>
			{
				int lngX = cXSegments > i + i ? i : cXSegments - i;//ie. i < cXSegments/2
				int lngY = cYSegments > j + j ? j : cYSegments - j;
				T weight = T.CreateTruncating((float)lngX / (lngX + lngY));
				return linearInterpolationX(unDst, i, j, dim1, dim2) * (T.One - weight) + linearInterpolationY(unDst, i, j, dim1, dim2) * weight;
			};
			GridIterator.iterate(dim1 - 1, dim2 - 1, (i, j) => unDst[i * dim2 + j] = func(i, j));
		}
	}
}

using System;
using System.Numerics;
using System.Threading.Tasks;

namespace VLP2D.Common
{
	public static class GridIterator
	{
		public static ParallelOptions optionsParallel;

		public static void iterate(int upperX, int upperY, Action<int, int> func)
		{
			Parallel.For(1, upperX, optionsParallel, i =>
			{
				for (int j = 1; j < upperY; j++) func(i, j);
			});
		}

		public static void iterate(int lowerX, int upperX, int lowerY, int upperY, Action<int, int> func)
		{
			Parallel.For(lowerX, upperX, optionsParallel, i =>
			{
				for (int j = lowerY; j < upperY; j++) func(i, j);
			});
		}

		public static void iterateWithEdges(int dim1, int dim2, Action<int, int> func)
		{
			Parallel.For(0, dim1, optionsParallel, i =>
			{
				for (int j = 0; j < dim2; j++) func(i, j);
			});
		}

		public static T iterateForMaxWithEps<T>(int upperX, int upperY, Action<int, int> func, Func<int, int, T> funcAbs, T eps) where T : INumber<T>, IMinMaxValue<T>
		{
			T deltaMax = T.MinValue;
			int cLoop = Math.Min(upperX, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMax = new T[cLoop];
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T deltaMaxColumn = deltaMax;//thread local maximum
				for (int i = 1 + core; i < upperX; i += cLoop)
				{
					for (int j = 1; j < upperY; j++)
					{
						func(i, j);
						if (deltaMaxColumn < eps)
						{
							T deltaVal = funcAbs(i, j);
							if (deltaMaxColumn < deltaVal) deltaMaxColumn = deltaVal;
						}
					}
				}
				fMax[core] = deltaMaxColumn;
			});
			for (int i = 0; i < cLoop; i++) deltaMax = T.Max(deltaMax, fMax[i]);
			return deltaMax;
		}

		public static bool iterateUntilCondition(int upperX, int upperY, Func<int, int, bool> funcBool)
		{
			bool rc = false;
			Parallel.For(1, upperX, optionsParallel, (i, loopState) =>
			{
				if (loopState.IsStopped) return;
				for (int j = 1; j < upperY; j++)
				{
					if (funcBool(i, j))
					{
						rc = true;
						loopState.Stop();
						return;
					}
				}
			});
			return rc;
		}

		static T iterateRedOrBlackForMax<T>(T[,] un, Action<int, int> func, Func<int, int, T> funcAbs, int val1, int val2, T eps) where T : INumber<T>, IMinMaxValue<T>
		{
			T deltaMax = T.MinValue;
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			int cLoop = Math.Min(upperX, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMax = new T[cLoop];
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T deltaMaxColumn = deltaMax;//thread local maximum
				for (int i = 1 + core; i < upperX; i += cLoop)
				{
					int initialIdx = ((i % 2) == 1) ? val1 : val2;
					for (int j = initialIdx; j < upperY; j += 2)
					{
						func(i, j);
						if (deltaMaxColumn < eps)
						{
							T deltaVal = funcAbs(i, j);
							if (deltaMaxColumn < deltaVal) deltaMaxColumn = deltaVal;
						}
					}
				}
				fMax[core] = deltaMaxColumn;
			});
			for (int i = 0; i < cLoop; i++) deltaMax = T.Max(deltaMax, fMax[i]);
			return deltaMax;
		}

		public static T iterateRedBlackForMax<T>(T[,] un, Action<int, int> funcForRedNodes, Action<int, int> funcForBlackNodes, Func<int, int, T> funcAbs, T eps) where T : INumber<T>, IMinMaxValue<T>
		{
			T deltaMax = iterateRedOrBlackForMax(un, funcForRedNodes, funcAbs, 1, 2, eps);
			if (deltaMax <= eps) deltaMax = iterateRedOrBlackForMax(un, funcForBlackNodes, funcAbs, 2, 1, eps);
			else iterateRedOrBlack(un, funcForBlackNodes, 2, 1);
			return deltaMax;
		}

		static T iterateRedOrBlackForMax<T>(T[,] un, Func<int, int, T> func, int val1, int val2, T eps) where T : INumber<T>, IMinMaxValue<T>
		{
			T deltaMax = T.MinValue;
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			int cLoop = Math.Min(upperX, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMax = new T[cLoop];
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T deltaMaxColumn = deltaMax;//thread local maximum
				for (int i = 1 + core; i < upperX; i += cLoop)
				{
					int initialIdx = ((i % 2) == 1) ? val1 : val2;
					for (int j = initialIdx; j < upperY; j += 2)
					{
						T prevVal = un[i, j];
						un[i, j] = func(i, j);
						if (deltaMaxColumn < eps)
						{
							T delta = un[i, j] - prevVal;
							T deltaVal = T.IsPositive(delta) ? delta : -delta;
							if (deltaMaxColumn < deltaVal) deltaMaxColumn = deltaVal;
						}
					}
				}
				fMax[core] = deltaMaxColumn;
			});
			for (int i = 0; i < cLoop; i++) deltaMax = T.Max(deltaMax, fMax[i]);
			return deltaMax;
		}

		public static T iterateRedBlackForMax<T>(T[,] un, Func<int, int, T> func, T eps) where T : INumber<T>, IMinMaxValue<T>
		{
			T deltaMax = iterateRedOrBlackForMax(un, func, 1, 2, eps);
			if (deltaMax <= eps) deltaMax = iterateRedOrBlackForMax(un, func, 2, 1, eps);
			else iterateRedOrBlack(un, func, 2, 1);
			return deltaMax;
		}

		static void iterateRedOrBlack<T>(T[,] un, Func<int, int, T> func, int val1, int val2)
		{
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			Parallel.For(1, upperX, optionsParallel, i =>
			{
				int initialIdx = ((i % 2) == 1) ? val1 : val2;
				for (int j = initialIdx; j < upperY; j += 2)
				{
					un[i, j] = func(i, j);
				}
			});
		}

		static void iterateRedOrBlack<T>(T[,] un, Action<int, int> func, int val1, int val2)
		{
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			Parallel.For(1, upperX, optionsParallel, i =>
			{
				int initialIdx = ((i % 2) == 1) ? val1 : val2;
				for (int j = initialIdx; j < upperY; j += 2)
				{
					func(i, j);
				}
			});
		}

		public static void iterateRedBlack<T>(T[,] un, Action<int, int> funcForRedNodes, Action<int, int> funcForBlackNodes) where T : INumber<T>
		{
			iterateRedOrBlack(un, funcForRedNodes, 1, 2);
			iterateRedOrBlack(un, funcForBlackNodes, 2, 1);
		}

		static void iterateRedSequent<T>(T[,] un, Action<int, int> func) where T : INumber<T>//for multigrid low dimensions
		{
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			for (int i = 1; i < upperX; i++)
			{
				int initialIdx = ((i % 2) == 1) ? 1 : 2;
				for (int j = initialIdx; j < upperY; j += 2)
				{
					func(i, j);
				}
			};
		}

		static void iterateBlackSequent<T>(T[,] un, Action<int, int> func) where T : INumber<T>//for multigrid low dimensions
		{
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			for (int i = 1; i < upperX; i++)
			{
				int initialIdx = ((i % 2) == 1) ? 2 : 1;
				for (int j = initialIdx; j < upperY; j += 2)
				{
					func(i, j);
				}
			};
		}

		public static void iterateRedBlackSequent<T>(T[,] un, Action<int, int> funcForRedNodes, Action<int, int> funcForBlackNodes) where T : INumber<T>
		{//for multigrid low dimensions
			iterateRedSequent(un, funcForRedNodes);
			iterateBlackSequent(un, funcForBlackNodes);
		}

		public static T iterateForSum<T>(T[,] un, Func<int, int, T> func, T[] columnSum) where T : INumber<T>
		{
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			Parallel.For(1, upperX, optionsParallel, i =>
			{
				columnSum[i] = T.Zero;
				for (int j = 1; j < upperY; j++)
				{
					columnSum[i] += func(i, j);
				}
			});
			T sum = T.Zero;
			for (int i = 1; i < upperX; i++) sum += columnSum[i];
			return sum;
		}

		public static T scalarProduct<T>(T[,] un, Func<int, int, T> func, T[] columnSum) where T : INumber<T>
		{
			return iterateForSum(un, func, columnSum);
		}

		public static void iterateSequent<T>(T[,] un, Action<int, int> func) where T : INumber<T>//for PTM
		{
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			for (int i = 1; i < upperX; i++)
			{
				for (int j = 1; j < upperY; j++)
				{
					func(i, j);
				}
			}
		}

		public static void iterateReverseSequent<T>(T[,] un, Action<int, int> func) where T : INumber<T>//for PTM
		{
			int upperX = un.GetUpperBound(0);
			int upperY = un.GetUpperBound(1);
			for (int i = upperX - 1; i >= 1; i--)
			{
				for (int j = upperY - 1; j >= 1; j--)
				{
					func(i, j);
				}
			}
		}

		public static void iterateWithIndeces(int parallelLoopCount, int initialJ, int stepJ, int upperY, ParallelOptions optionsPara, Action<int, int> action)
		{
			Parallel.For(0, parallelLoopCount, optionsPara, j =>
			{
				for (int i = 1; i <= upperY; i++) action(initialJ + j * stepJ, i);
			});
		}

		public static void iterateEdgesAndFillInternalPoints(int dim1, int dim2, Action<int, int> actionBoundary, Action<int, int> actionInternal)
		{
			Parallel.For(0, dim1, optionsParallel, i =>
			{
				for (int j = 0; j < dim2; j++)
				{
					if (i > 0 && i < dim1 - 1 && j > 0 && j < dim2 - 1)
					{
						actionInternal?.Invoke(i, j);
					}
					else actionBoundary?.Invoke(i, j);
				}
			});
		}
	}
}

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using static VLP2D.Common.Utils;

namespace VLP2D.Common
{
	internal class UtilsDiff
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void updateMinMax<T>(T val, ref T fMin, ref T fMax) where T : INumber<T>
		{
			if (fMin > val) fMin = val;
			if (fMax < val) fMax = val;
		}

		public static void pointsMinMax<T>(Adapter2D<T> adapter, ref T valMin, ref T valMax) where T : INumber<T>, IMinMaxValue<T>
		{
			int cLoop = Math.Min(adapter.dim1 - 1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, T.MaxValue);
			Array.Fill(fMax, T.MinValue);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				for (int i = 0 + core; i < adapter.dim1; i += cLoop)
				{
					for (int j = 0; j < adapter.dim2; j++)
					{
						updateMinMax(adapter.func(i, j), ref fMin[core], ref fMax[core]);
					}
				}
			});
			for (int i = 0; i < cLoop; i++)
			{
				valMin = T.Min(fMin[i], valMin);
				valMax = T.Max(fMax[i], valMax);
			}
		}

		public static void calculateDifference<T>(Adapter2D<T> adapter, T[][] unDiff, T stpX, T stpY, Func<T, T, T> funcAnalitic, ref T valMin, ref T valMax, Func<bool> canceled, Action<double> reportProgress) where T : INumber<T>, IMinMaxValue<T>//, IMultiplyOperators<T, double, T>
		{
			ulong progressSteps = (ulong)((adapter.dim1 - 1) * (adapter.dim2 - 1)), curProgress = 0;//UInt64
			reportProgress?.Invoke(0);

			int cLoop = Math.Min(adapter.dim1 - 1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T[] fMin = new T[cLoop], fMax = new T[cLoop];
			Array.Fill(fMin, T.MaxValue);
			Array.Fill(fMax, T.MinValue);
			T incX = stpX * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				T x = stpX * T.CreateTruncating(1 + core);
				for (int i = 1 + core; i < adapter.dim1; i += cLoop)
				{
					T y = T.Zero;
					for (int j = 1; j < adapter.dim2; j++)
					{
						y += stpY;
						T val = adapter.func(i, j);
						if (!T.IsNaN(val))
						{
							T diff = adapter.func(i, j) - funcAnalitic(x, y);
							unDiff[i][j] = diff;
							updateMinMax(diff, ref fMin[core], ref fMax[core]);
						}
						else unDiff[i][j] = val;
						if ((canceled != null) && (j % 10 == 0) && canceled()) goto exitLoop;
					}
					if (reportProgress != null)
					{
						Interlocked.Add(ref curProgress, (ulong)(adapter.dim2 - 1));
						if (i % 10 == 0) reportProgress(curProgress * 100.0 / progressSteps);
					}
				exitLoop:;
					x += incX;
					if ((canceled != null) && canceled()) break;
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
	}
}

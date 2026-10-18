using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using static VLP2D.Common.Utils;

namespace VLP2D.Common
{
	internal class UtilsDiff<T> where T : INumber<T>, IMinMaxValue<T>
	{
		static T[] fMin = new T[Environment.ProcessorCount], fMax = new T[Environment.ProcessorCount];

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void updateMinMax(T val, ref T fMin, ref T fMax)
		{
			if (fMin > val) fMin = val;
			if (fMax < val) fMax = val;
		}

		public static void pointsMinMax(Adapter2D<T> adapter, ref T valMin, ref T valMax)
		{
			int cLoop = Math.Min(adapter.dim1 - 1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				fMin[core] = T.MaxValue;
				fMax[core] = T.MinValue;
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

		public static void calculateDifference(Adapter2D<T> adapter, T[][] unDiff, T xMin, T yMin, T stpX, T stpY, Func<T, T, T> funcAnalitic, ref T valMin, ref T valMax, Func<bool> canceled, Action<double> reportProgress)
		{
			ulong progressSteps = (ulong)((adapter.dim1 - 1) * (adapter.dim2 - 1)), curProgress = 0;//UInt64
			reportProgress?.Invoke(0);

			int cLoop = Math.Min(adapter.dim1 - 1, GridIterator.optionsParallel.MaxDegreeOfParallelism);
			T incX = stpX * T.CreateTruncating(cLoop);
			Parallel.For(0, cLoop, GridIterator.optionsParallel, (core) =>
			{
				fMin[core] = T.MaxValue;
				fMax[core] = T.MinValue;
				T x = xMin + stpX * T.CreateTruncating(1 + core);
				for (int i = 1 + core; i < adapter.dim1; i += cLoop)
				{
					T y = yMin;
					for (int j = 1; j < adapter.dim2; j++)
					{
						y += stpY;
						T val = adapter.func(i, j);
						if (!T.IsNaN(val))
						{
							T valAnalitic = funcAnalitic(x, y);
							if (!T.IsNaN(valAnalitic))
							{
								T diff = adapter.func(i, j) - valAnalitic;
								unDiff[i][j] = diff;
								updateMinMax(diff, ref fMin[core], ref fMax[core]);
							}
							else unDiff[i][j] = valAnalitic;
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

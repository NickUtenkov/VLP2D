using System.Numerics;
using System.Runtime.CompilerServices;

namespace VLP2D.Common
{
	internal class UtilsOpLap
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static T operatorLaplace<T>(T[,] u, int i, int j, T _4) where T : IAdditionOperators<T, T, T>, ISubtractionOperators<T, T, T>, IMultiplyOperators<T, T, T>
		{//using without dividing by step2(not needed in some cases)
			return u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - u[i, j] * _4;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static T operatorLaplaceXY<T>(T[,] u, int i, int j, T stepX2, T stepY2, T _2) where T : IAdditionOperators<T, T, T>, ISubtractionOperators<T, T, T>, IDivisionOperators<T, T, T>, IMultiplyOperators<T, T, T>
			//, IMultiplyOperators<T, double, T>
		{
			T u2 = u[i, j] * _2;
			return (u[i - 1, j] + u[i + 1, j] - u2) / stepX2 + (u[i, j - 1] + u[i, j + 1] - u2) / stepY2;
		}
	}
}

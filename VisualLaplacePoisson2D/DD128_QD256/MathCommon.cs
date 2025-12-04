using System;
using System.Numerics;
using UtilsCommon;

namespace MathematicCommon
{
	internal class MathCommon
	{
		public static T npwr<T>(T a, int n) where T : INumber<T>
		{
			if (n == 0)
			{
				if (T.IsZero(a))
				{
					Utils.Error("npwr: Invalid argument.");
					return T.CreateTruncating(double.NaN);
				}
				return T.One;
			}

			T r = a;
			T s = T.One;
			int N = Math.Abs(n);

			if (N > 1)
			{
				// Use binary exponentiation
				while (N > 0)
				{
					if (N % 2 == 1) s *= r;
					N /= 2;
					if (N > 0) r *= r;
				}
			}
			else s = r;

			if (n < 0) return T.One / s;// Compute the reciprocal if n is negative.

			return s;
		}

		// Computes the nearest integer to d
		public static double nint(double d)
		{
			if (d == Math.Floor(d)) return d;
			return Math.Floor(d + 0.5);
		}
	}
}

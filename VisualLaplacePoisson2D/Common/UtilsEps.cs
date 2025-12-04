using DD128Numeric;
using QD256Numeric;
using System.Numerics;

namespace VLP2D.Common
{
	internal class UtilsEps
	{
		public static T machineEpsilon<T>() where T : INumber<T>
		{
			T e = T.One;
			T _2 = T.CreateTruncating(2.0);

			int count = 0;
			while (T.One + e / _2 > T.One)
			{
				e /= _2;
				count++;
			}

			return e;
		}

		public static double epsilon<T>()
		{
			if (typeof(T) == typeof(float)) return 1.1920929E-07;
			if (typeof(T) == typeof(double)) return 2.2204460492503131E-16;
			if (typeof(T) == typeof(DD128)) return 4.93038065763132e-32;
			if (typeof(T) == typeof(QD256)) return 1.21543267145725e-63;

			return 2.2204460492503131E-16;
		}
	}
}

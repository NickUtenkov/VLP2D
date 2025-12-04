using System.Runtime.CompilerServices;

namespace DD128Numeric
{
	internal static class ArithmeticDD128
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static (double high, double low) Split(double a)
		{
			const double _QD_SPLITTER = 134217729.0;// = 2^27 + 1
			const double _QD_SPLIT_THRESH = 6.69692879491417e+299;// = 2^996
			double hi, lo, temp;
			if (a > _QD_SPLIT_THRESH || a < -_QD_SPLIT_THRESH)
			{
				a *= 3.7252902984619140625e-09;  // 2^-28
				temp = _QD_SPLITTER * a;
				hi = temp - (temp - a);
				lo = a - hi;
				hi *= 268435456.0;          // 2^28
				lo *= 268435456.0;          // 2^28
			}
			else
			{
				temp = _QD_SPLITTER * a;
				hi = temp - (temp - a);
				lo = a - hi;
			}
			return (hi, lo);
		}
		#region Additions

		/// <summary>
		/// Computes sum of two doubles and returns it as instance of <see cref="DD128"/>.
		/// </summary>
		/// <param name="a">First argument.</param>
		/// <param name="b">Second argument.</param>
		/// <returns></returns>
		public static DD128 QuickTwoSum(double a, double b)
		{
			double sum = a + b;
			return new DD128(sum, b - (sum - a));
		}

		public static double QuickTwoSum(double a, double b, out double err)
		{
			double sum = a + b;
			err = b - (sum - a);
			return sum;
		}

		/// <summary>
		/// Computes sum of two doubles and returns it as instance of <see cref="DD128"/>.
		/// </summary>
		/// <param name="a">First argument.</param>
		/// <param name="b">Second argument.</param>
		/// <returns></returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 TwoSum(double a, double b)
		{
			var sum = a + b;
			double bb = sum - a;
			return new DD128(sum, (a - (sum - bb)) + (b - bb));
		}
		/* Computes fl(a+b) and err(a+b).  */
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double TwoSum(double a, double b, out double err)
		{
			double sum = a + b;
			double bb = sum - a;
			err = (a - (sum - bb)) + (b - bb);
			return sum;
		}

		#endregion

		#region Subtractions

		/// <summary>
		/// Computes difference of two doubles and associated error assuming that |a| >= |b|.
		/// </summary>
		/// <param name="a">First argument.</param>
		/// <param name="b">Second argument.</param>
		/// <returns></returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static (double diff, double error) QuickTwoDiff(double a, double b)
		{
			double diff = a - b;
			return (diff, (a - diff) - b);
		}

		/// <summary>
		/// Computes difference of two doubles and associated error. 
		/// </summary>
		/// <param name="a">First argument.</param>
		/// <param name="b">Second argument.</param>
		/// <returns></returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static (double sum, double error) TwoDiff(double a, double b)
		{
			var diff = a - b;
			double bb = diff - a;
			return (diff, (a - (diff - bb)) - (b + bb));
		}

		#endregion

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static (double product, double error) TwoProd(double a, double b)
		{
			// TODO: use FMA instructions instead of the Split solution
			double product = a * b;
			(double ahi, double alo) = Split(a);
			(double bhi, double blo) = Split(b);
			return (product, ((ahi * bhi - product) + ahi * blo + alo * bhi) + alo * blo);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double TwoSqr(double a, out double err)
		{
#if HAS_FMA
			double p = -a * a;
			err = a * a + p;
			return -p;
#else
			double hi, lo;
			double q = a * a;
			(hi, lo) = Split(a);
			err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
			return q;
#endif
		}
	}
}

using DD128Numeric;
using System.Runtime.CompilerServices;

namespace QD256Numeric
{
	internal class ArithmeticQD256
	{
		public static QD256 Renormalize(double c0, double c1, double c2, double c3, double c4)
		{
			if (double.IsInfinity(c0)) return new QD256(c0, c1, c2, c3);

			double s0;
			(s0, c4) = ArithmeticDD128.QuickTwoSum(c3, c4);
			(s0, c3) = ArithmeticDD128.QuickTwoSum(c2, s0);
			(s0, c2) = ArithmeticDD128.QuickTwoSum(c1, s0);
			(c0, c1) = ArithmeticDD128.QuickTwoSum(c0, s0);

			s0 = c0;
			var s1 = c1;
			double s2 = 0;
			double s3 = 0;

			(s0, s1) = ArithmeticDD128.QuickTwoSum(c0, c1);
			if (s1 != 0.0)
			{
				(s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c2);
				if (s2 != 0.0)
				{
					(s2, s3) = ArithmeticDD128.QuickTwoSum(s2, c3);
					if (s3 != 0.0) s3 += c4;
					else s2 += c4;
				}
				else
				{
					(s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c3);
					if (s2 != 0.0) (s2, s3) = ArithmeticDD128.QuickTwoSum(s2, c4);
					else (s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c4);
				}
			}
			else
			{
				(s0, s1) = ArithmeticDD128.QuickTwoSum(s0, c2);
				if (s1 != 0.0)
				{
					(s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c3);
					if (s2 != 0.0) (s2, s3) = ArithmeticDD128.QuickTwoSum(s2, c4);
					else (s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c4);
				}
				else
				{
					(s0, s1) = ArithmeticDD128.QuickTwoSum(s0, c3);
					if (s1 != 0.0) (s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c4);
					else (s0, s1) = ArithmeticDD128.QuickTwoSum(s0, c4);
				}
			}

			return new QD256(s0, s1, s2, s3);
		}

		public static QD256 Renormalize(double c0, double c1, double c2, double c3)
		{
			if (double.IsInfinity(c0)) return new QD256(c0, c1, c2, c3);

			double s0;
			(s0, c3) = ArithmeticDD128.QuickTwoSum(c2, c3);
			(s0, c2) = ArithmeticDD128.QuickTwoSum(c1, s0);
			(c0, c1) = ArithmeticDD128.QuickTwoSum(c0, s0);

			s0 = c0;
			var s1 = c1;
			double s2 = 0;
			double s3 = 0;
			if (s1 != 0.0)
			{
				(s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c2);
				if (s2 != 0.0) (s2, s3) = ArithmeticDD128.QuickTwoSum(s2, c3);
				else (s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c3);
			}
			else
			{
				(s0, s1) = ArithmeticDD128.QuickTwoSum(s0, c2);
				if (s1 != 0.0) (s1, s2) = ArithmeticDD128.QuickTwoSum(s1, c3);
				else (s0, s1) = ArithmeticDD128.QuickTwoSum(s0, c3);
			}

			return new QD256(s0, s1, s2, s3);
		}

		public static QD256 RenormalizeQuick(double c0, double c1, double c2, double c3, double c4)
		{
			double t0, t1, t2, t3;
			double s;
			(s, t3) = ArithmeticDD128.QuickTwoSum(c3, c4);
			(s, t2) = ArithmeticDD128.QuickTwoSum(c2, s);
			(s, t1) = ArithmeticDD128.QuickTwoSum(c1, s);
			(c0, t0) = ArithmeticDD128.QuickTwoSum(c0, s);

			(s, t2) = ArithmeticDD128.QuickTwoSum(t2, t3);
			(s, t1) = ArithmeticDD128.QuickTwoSum(t1, s);
			(c1, t0) = ArithmeticDD128.QuickTwoSum(t0, s);

			(s, t1) = ArithmeticDD128.QuickTwoSum(t1, t2);
			(c2, t0) = ArithmeticDD128.QuickTwoSum(t0, s);

			c3 = t0 + t1;

			return new QD256(c0, c1, c2, c3);
		}

		public static void ThreeSum(ref double a, ref double b, ref double c)
		{
			double t1, t2, t3;
			(t1, t2) = ArithmeticDD128.TwoSum(a, b);
			(a, t3) = ArithmeticDD128.TwoSum(c, t1);
			(b, c) = ArithmeticDD128.TwoSum(t2, t3);
		}

		public static void ThreeSum2(ref double a, ref double b, ref double c)
		{
			double t1, t2, t3;
			(t1, t2) = ArithmeticDD128.TwoSum(a, b);
			(a, t3) = ArithmeticDD128.TwoSum(c, t1);
			b = t2 + t3;
		}

		public static double QuickThreeAccum(ref double a, ref double b, double c)
		{
			double s;
			bool za, zb;

			(s, b) = ArithmeticDD128.TwoSum(b, c);
			(s, a) = ArithmeticDD128.TwoSum(a, s);

			za = a != 0.0;
			zb = b != 0.0;

			if (za & zb) return s;

			if (!zb)
			{
				b = a;
				a = s;
			}
			else a = s;

			return 0.0;
		}
	}
}

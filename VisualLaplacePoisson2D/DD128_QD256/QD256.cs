#define UseDD128

using DD128Numeric;
using MathematicCommon;
using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using UtilsCommon;

namespace QD256Numeric
{
	/// <summary>
	///     Represents a floating number with quad-double precision (209-bits)
	/// </summary>
	public struct QD256 : IConvertible, INumber<QD256>, ILogarithmicFunctions<QD256>, IPowerFunctions<QD256>, IExponentialFunctions<QD256>,
		IMinMaxValue<QD256>, IRootFunctions<QD256>, ITrigonometricFunctions<QD256>, IHyperbolicFunctions<QD256>, IFloatingPointConstants<QD256>
		, IAdditionOperators<QD256, double, QD256>, ISubtractionOperators<QD256, double, QD256>
		, IMultiplyOperators<QD256, double, QD256>, IDivisionOperators<QD256, double, QD256>
		, IAdditionOperators<QD256, float, QD256>, ISubtractionOperators<QD256, float, QD256>
		, IMultiplyOperators<QD256, float, QD256>, IDivisionOperators<QD256, float, QD256>
	{
		internal readonly double x0;
		internal readonly double x1;
		internal readonly double x2;
		internal readonly double x3;

		const int digits = 209;
		public const int digits10 = 62;
		public const double PositiveInfinity = (double)1.0 / (double)0.0;
		public static QD256 Zero => MathQD256._zero;
		public static QD256 One => MathQD256._one;
		public const double Epsilon = MathQD256._eps;
		public static QD256 E = MathQD256._e;
		public static QD256 Pi = MathQD256._pi;
		public static QD256 MinValue = -MathQD256._max;
		public static QD256 MaxValue = MathQD256._max;

		public QD256(double x0, double x1 = 0, double x2 = 0, double x3 = 0)
		{
			this.x0 = x0;
			this.x1 = x1;
			this.x2 = x2;
			this.x3 = x3;
		}

		#region Addition

		  public static QD256 operator +(QD256 left, QD256 right)
		  {
				int i, j, k;
				double s, t;
				double u, v;
				Span<double> x = stackalloc double[4];

				i = j = k = 0;
				if (Math.Abs(left.x0) > Math.Abs(right.x0)) u = left[i++];
				else u = right[j++];

				if (Math.Abs(left[i]) > Math.Abs(right[j])) v = left[i++];
				else v = right[j++];

				(u, v) = ArithmeticDD128.QuickTwoSum(u, v);

				while (k < 4)
				{
					 if (i >= 4 && j >= 4)
					 {
						  x[k] = u;
						  if (k < 3) x[++k] = v;
						  break;
					 }

					 if (i >= 4) t = right[j++];
					 else if (j >= 4) t = left[i++];
					 else if (Math.Abs(left[i]) > Math.Abs(right[j])) t = left[i++];
					 else t = right[j++];

					 s = ArithmeticQD256.QuickThreeAccum(ref u, ref v, t);

					 if (s != 0.0) x[k++] = s;
				}

				for (k = i; k < 4; k++) x[3] += left[k];
				for (k = j; k < 4; k++) x[3] += right[k];

				return ArithmeticQD256.Renormalize(x[0], x[1], x[2], x[3]);
		  }

#if UseDD128
		public static QD256 operator +(QD256 left, DD128 right)
		  {
				var (s0, t0) = ArithmeticDD128.TwoSum(left.x0, right.x0);
				var (s1, t1) = ArithmeticDD128.TwoSum(left.x1, right.x1);

				(s1, t0) = ArithmeticDD128.TwoSum(s1, t0);

				var s2 = left.x2;
				ArithmeticQD256.ThreeSum(ref s2, ref t0, ref t1);

				double s3;
				(s3, t0) = ArithmeticDD128.TwoSum(t0, left.x3);
				t0 += t1;

				return ArithmeticQD256.Renormalize(s0, s1, s2, s3, t0);
		  }
#endif
#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator +(DD128 left, QD256 right) => right + left;
#endif
		public static QD256 operator +(QD256 left, double right)
		{
				double c0, c1, c2, c3;
				double e;

				(c0, e) = ArithmeticDD128.TwoSum(left.x0, right);
				(c1, e) = ArithmeticDD128.TwoSum(left.x1, e);
				(c2, e) = ArithmeticDD128.TwoSum(left.x2, e);
				(c3, e) = ArithmeticDD128.TwoSum(left.x3, e);

				return ArithmeticQD256.Renormalize(c0, c1, c2, c3, e);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator +(double left, QD256 right) => right + left;

		public static QD256 operator +(QD256 left, float right) => left + right;
		public static QD256 operator +(float left, QD256 right) => right + left;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 Add(double left, double right) => new QD256(left) + right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public QD256 Add(QD256 other) => this + other;

#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 Add(DD128 other) => this + other;
#endif
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 Add(double other) => this + other;

		  public QD256 AddSloppy(QD256 other)
		  {
				double s0, s1, s2, s3;
				double t0, t1, t2, t3;

				(s0, t0) = ArithmeticDD128.TwoSum(x0, other.x0);
				(s1, t1) = ArithmeticDD128.TwoSum(x1, other.x1);
				(s2, t2) = ArithmeticDD128.TwoSum(x2, other.x2);
				(s3, t3) = ArithmeticDD128.TwoSum(x3, other.x3);

				(s1, t0) = ArithmeticDD128.TwoSum(s1, t0);
				ArithmeticQD256.ThreeSum(ref s2, ref t0, ref t1);
				ArithmeticQD256.ThreeSum2(ref s3, ref t0, ref t2);
				t0 = t0 + t1 + t3;

				return ArithmeticQD256.Renormalize(s0, s1, s2, s3, t0);
		  }

		  #endregion

		#region Subtraction

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator -(QD256 value) => new QD256(-value.x0, -value.x1, -value.x2, -value.x3);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator -(QD256 left, QD256 right) => left + -right;

#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator -(QD256 left, DD128 right) => left + -right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator -(DD128 left, QD256 right) => left + -right;
#endif
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator -(QD256 left, double right) => left + -right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator -(double left, QD256 right) => left + -right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator -(QD256 left, float right) => left + -right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator -(float left, QD256 right) => (QD256)left + -right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 Subtract(double left, double right) => new QD256(left) - right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public QD256 Subtract(QD256 other) => this - other;

#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 Subtract(DD128 other) => this - other;
#endif
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 Subtract(double other) => this - other;

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 SubtractSloppy(QD256 other) => AddSloppy(-other);
			
		#endregion

		#region Multiplication

		public static QD256 operator *(QD256 left, QD256 right)
		{
			double p0, p1, p2, p3, p4, p5;
			double q0, q1, q2, q3, q4, q5;
			double p6, p7, p8, p9;
			double q6, q7, q8, q9;
			double r0, r1;
			double t0, t1;
			double s0, s1, s2;

			(p0, q0) = ArithmeticDD128.TwoProd(left.x0, right.x0);

			(p1, q1) = ArithmeticDD128.TwoProd(left.x0, right.x1);
			(p2, q2) = ArithmeticDD128.TwoProd(left.x1, right.x0);

			(p3, q3) = ArithmeticDD128.TwoProd(left.x0, right.x2);
			(p4, q4) = ArithmeticDD128.TwoProd(left.x1, right.x1);
			(p5, q5) = ArithmeticDD128.TwoProd(left.x2, right.x0);

			/* Start Accumulation */
			ArithmeticQD256.ThreeSum(ref p1, ref p2, ref q0);

			/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
			ArithmeticQD256.ThreeSum(ref p2, ref q1, ref q2);
			ArithmeticQD256.ThreeSum(ref p3, ref p4, ref p5);
			/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
			(s0, t0) = ArithmeticDD128.TwoSum(p2, p3);
			(s1, t1) = ArithmeticDD128.TwoSum(q1, p4);
			s2 = q2 + p5;
			(s1, t0) = ArithmeticDD128.TwoSum(s1, t0);
			s2 += t0 + t1;

			/* O(eps^3) order terms */
			(p6, q6) = ArithmeticDD128.TwoProd(left.x0, right.x3);
			(p7, q7) = ArithmeticDD128.TwoProd(left.x1, right.x2);
			(p8, q8) = ArithmeticDD128.TwoProd(left.x2, right.x1);
			(p9, q9) = ArithmeticDD128.TwoProd(left.x3, right.x0);

			/* Nine-Two-Sum of q0, s1, q3, q4, q5, p6, p7, p8, p9. */
			(q0, q3) = ArithmeticDD128.TwoSum(q0, q3);
			(q4, q5) = ArithmeticDD128.TwoSum(q4, q5);
			(p6, p7) = ArithmeticDD128.TwoSum(p6, p7);
			(p8, p9) = ArithmeticDD128.TwoSum(p8, p9);
			/* Compute (t0, t1) = (q0, q3) + (q4, q5). */
			(t0, t1) = ArithmeticDD128.TwoSum(q0, q4);
			t1 += q3 + q5;
			/* Compute (r0, r1) = (p6, p7) + (p8, p9). */
			(r0, r1) = ArithmeticDD128.TwoSum(p6, p8);
			r1 += p7 + p9;
			/* Compute (q3, q4) = (t0, t1) + (r0, r1). */
			(q3, q4) = ArithmeticDD128.TwoSum(t0, r0);
			q4 += t1 + r1;
			/* Compute (t0, t1) = (q3, q4) + s1. */
			(t0, t1) = ArithmeticDD128.TwoSum(q3, s1);
			t1 += q4;

			/* O(eps^4) terms -- Nine-One-Sum */
			t1 += left.x1 * right.x3 +
					left.x2 * right.x2 +
					left.x3 * right.x1 +
					q6 +
					q7 +
					q8 +
					q9 +
					s2;

			return ArithmeticQD256.Renormalize(p0, p1, s0, t0, t1);
		}

#if UseDD128
		public static QD256 operator *(QD256 left, DD128 right)
		  {
				double p0, p1, p2, p3, p4;
				double q0, q1, q2, q3, q4;
				double s0, s1, s2;
				double t0, t1;

				(p0, q0) = ArithmeticDD128.TwoProd(left.x0, right.x0);
				(p1, q1) = ArithmeticDD128.TwoProd(left.x0, right.x1);
				(p2, q2) = ArithmeticDD128.TwoProd(left.x1, right.x0);
				(p3, q3) = ArithmeticDD128.TwoProd(left.x1, right.x1);
				(p4, q4) = ArithmeticDD128.TwoProd(left.x2, right.x0);

				ArithmeticQD256.ThreeSum(ref p1, ref p2, ref q0);

				ArithmeticQD256.ThreeSum(ref p2, ref p3, ref p4);
				(q1, q2) = ArithmeticDD128.TwoSum(q1, q2);
				(s0, t0) = ArithmeticDD128.TwoSum(p2, q1);
				(s1, t1) = ArithmeticDD128.TwoSum(p3, q2);
				(s1, t0) = ArithmeticDD128.TwoSum(s1, t0);
				s2 = t0 + t1 + p4;
				p2 = s0;

				p2 = left.x2 * right.x0 + left.x3 * right.x1 + q3 + q4;
				ArithmeticQD256.ThreeSum2(ref p3, ref q0, ref s1);
				p4 = q0 + s2;

				return ArithmeticQD256.Renormalize(p0, p1, p2, p3);
		  }
#endif
#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator *(DD128 left, QD256 right) => right * left;
#endif
		public static QD256 operator *(QD256 left, double right)
		{
				double p0, p1, p2, p3;
				double q0, q1, q2;
				double s0, s1, s2, s3, s4;

				(p0, q0) = ArithmeticDD128.TwoProd(left.x0, right);
				(p1, q1) = ArithmeticDD128.TwoProd(left.x1, right);
				(p2, q2) = ArithmeticDD128.TwoProd(left.x2, right);
				p3 = left.x3 * right;

				s0 = p0;

				(s1, s2) = ArithmeticDD128.TwoSum(q0, p1);

				ArithmeticQD256.ThreeSum(ref s2, ref q1, ref p2);
				ArithmeticQD256.ThreeSum2(ref q1, ref q2, ref p3);
				s3 = q1;
				s4 = q2 + p2;

				return ArithmeticQD256.Renormalize(s0, s1, s2, s3, s4);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator *(double left, QD256 right) => right * left;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator *(QD256 left, float right) => left * (QD256)right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator *(float left, QD256 right) => (QD256)left * right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 Multiply(double left, double right) => new QD256(left) * right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public QD256 Multiply(QD256 other) => this * other;

#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 Multiply(DD128 other) => this * other;
#endif
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 Multiply(double other) => this * other;

		  public QD256 MultiplySloppy(QD256 other)
		  {
				double p0, p1, p2, p3, p4, p5;
				double q0, q1, q2, q3, q4, q5;
				double t0, t1;
				double s0, s1, s2;

				(p0, q0) = ArithmeticDD128.TwoProd(x0, other.x0);

				(p1, q1) = ArithmeticDD128.TwoProd(x0, other.x1);
				(p2, q2) = ArithmeticDD128.TwoProd(x1, other.x0);

				(p3, q3) = ArithmeticDD128.TwoProd(x0, other.x2);
				(p4, q4) = ArithmeticDD128.TwoProd(x1, other.x1);
				(p5, q5) = ArithmeticDD128.TwoProd(x2, other.x0);

				/* Start Accumulation */
				ArithmeticQD256.ThreeSum(ref p1, ref p2, ref q0);

				/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
				ArithmeticQD256.ThreeSum(ref p2, ref q1, ref q2);
				ArithmeticQD256.ThreeSum(ref p3, ref p4, ref p5);
				/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
				(s0, t0) = ArithmeticDD128.TwoSum(p2, p3);
				(s1, t1) = ArithmeticDD128.TwoSum(q1, p4);
				s2 = q2 + p5;
				(s1, t0) = ArithmeticDD128.TwoSum(s1, t0);
				s2 += t0 + t1;

				/* O(eps^3) order terms */
				s1 += x0 * other.x3 + x1 * other.x2 + x2 * other.x1 + x3 * other.x0 + q0 + q3 + q4 + q5;
				return ArithmeticQD256.Renormalize(p0, p1, s0, s1, s2);
		  }

		  #endregion

		#region Division

		  public static QD256 operator /(QD256 left, QD256 right)
		  {
				double q0, q1, q2, q3;

				QD256 r;

				q0 = left.x0 / right.x0;
				r = left - right * q0;

				q1 = r.x0 / right.x0;
				r -= right * q1;

				q2 = r.x0 / right.x0;
				r -= right * q2;

				q3 = r.x0 / right.x0;
				r -= right * q3;

				var q4 = r.x0 / right.x0;

				return ArithmeticQD256.Renormalize(q0, q1, q2, q3, q4);
		  }

#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator /(QD256 left, DD128 right) => left / (QD256) right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator /(DD128 left, QD256 right) => (QD256) left / right;
#endif
#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static QD256 operator /(QD256 left, double right)
		  {
				/* Strategy:  compute approximate quotient using high order
					doubles, and then correct it 3 times using the remainder.
					(Analogous to long division.)                             */
				double t0, t1;
				double q0, q1, q2, q3;
				QD256 r;

				q0 = left.x0 / right; /* approximate quotient */

				/* Compute the remainder  a - q0 * right */
				(t0, t1) = ArithmeticDD128.TwoProd(q0, right);
				r = left - new DD128(t0, t1);

				/* Compute the first correction */
				q1 = r[0] / right;
				(t0, t1) = ArithmeticDD128.TwoProd(q1, right);
				r -= new DD128(t0, t1);

				/* Second correction to the quotient. */
				q2 = r[0] / right;
				(t0, t1) = ArithmeticDD128.TwoProd(q2, right);
				r -= new QD256(t0, t1);

				/* Final correction to the quotient. */
				q3 = r[0] / right;

				return ArithmeticQD256.Renormalize(q0, q1, q2, q3);
		  }
#endif

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator /(double left, QD256 right) => (QD256) left / right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator /(QD256 left, float right) => left / (QD256)right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 operator /(float left, QD256 right) => (QD256)left / right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 Divide(double left, double right) => new QD256(left) / right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public QD256 Divide(QD256 other) => this / other;

#if UseDD128
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 Divide(DD128 other) => this / other;
#endif
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public QD256 Divide(double other) => this / other;

		  public QD256 DivideSloppy(QD256 other)
		  {
				double q0, q1, q2, q3;

				QD256 r;

				q0 = x0 / other.x0;
				r = this - other * q0;

				q1 = r[0] / other.x0;
				r -= other * q1;

				q2 = r[0] / other.x0;
				r -= other * q2;

				q3 = r[0] / other.x0;

				return ArithmeticQD256.Renormalize(q0, q1, q2, q3);
		  }

		#endregion

		#region Other math operators

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool IsNaN(QD256 qd)
		  {
				return double.IsNaN(qd.x0) | double.IsNaN(qd.x1) | double.IsNaN(qd.x2) | double.IsNaN(qd.x3);
		  }

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool IsInfinity(QD256 qd)
		  {
				return double.IsInfinity(qd.x0);
		  }

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool IsFinite(QD256 qd)
		  {
				return !(double.IsInfinity(qd.x0) | double.IsNaN(qd.x0));
		  }

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool IsPositiveInfinity(QD256 qd)
		  {
				return double.IsPositiveInfinity(qd.x0);
		  }

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool IsNegativeInfinity(QD256 qd)
		  {
				return double.IsNegativeInfinity(qd.x0);
		  }
		#endregion

		#region Conversions

		// To QD256

#if UseDD128
		public static implicit operator QD256(DD128 value) => new QD256(value.x0, value.x1);
#endif
		// TODO: dont throw away the truncated data
		public static explicit operator QD256(decimal value) => new QD256((double) value);

		  public static implicit operator QD256(double value) => new QD256(value);

		  public static implicit operator QD256(float value) => new QD256(value);

		  public static implicit operator QD256(ulong value) => new QD256(value);

		  public static implicit operator QD256(long value) => new QD256(value);

		  public static implicit operator QD256(uint value) => new QD256(value);

		  public static implicit operator QD256(int value) => new QD256(value);

		  public static implicit operator QD256(short value) => new QD256(value);

		  public static implicit operator QD256(ushort value) => new QD256(value);

		  public static implicit operator QD256(byte value) => new QD256(value);

		  public static implicit operator QD256(sbyte value) => new QD256(value);

		  public static implicit operator QD256(char value) => new QD256(value);

		  // From QD256

		  public static explicit operator double(QD256 value) => value.x0;

		  public static explicit operator decimal(QD256 value)
				=> (decimal)value.x0 + (decimal)value.x1 + (decimal)value.x2 + (decimal)value.x3;

		  public static explicit operator float(QD256 value) => (float) value.x0;

		  public static explicit operator ulong(QD256 value) => (ulong) value.x0;

		  public static explicit operator long(QD256 value) => (long) value.x0;

		  public static explicit operator uint(QD256 value) => (uint) value.x0;

		  public static explicit operator int(QD256 value) => (int) value.x0;

		  public static explicit operator short(QD256 value) => (short) value.x0;

		  public static explicit operator ushort(QD256 value) => (ushort) value.x0;

		  public static explicit operator byte(QD256 value) => (byte) value.x0;

		  public static explicit operator sbyte(QD256 value) => (sbyte) value.x0;

		  public static explicit operator char(QD256 value) => (char) value.x0;

		  TypeCode IConvertible.GetTypeCode() => TypeCode.Double;

		  bool IConvertible.ToBoolean(IFormatProvider provider)
				=> throw new InvalidCastException("Cannot cast QD256 to bool");

		  byte IConvertible.ToByte(IFormatProvider provider) => (byte) this;

		  char IConvertible.ToChar(IFormatProvider provider) => (char) this;

		  DateTime IConvertible.ToDateTime(IFormatProvider provider)
				=> throw new InvalidCastException("Cannot cast QD256 to DateTime");

		  decimal IConvertible.ToDecimal(IFormatProvider provider) => (decimal) this;

		  double IConvertible.ToDouble(IFormatProvider provider) => (double) this;

		  short IConvertible.ToInt16(IFormatProvider provider) => (short) this;

		  int IConvertible.ToInt32(IFormatProvider provider) => (int) this;

		  long IConvertible.ToInt64(IFormatProvider provider) => (long) this;

		  sbyte IConvertible.ToSByte(IFormatProvider provider) => (sbyte) this;

		  float IConvertible.ToSingle(IFormatProvider provider) => (float) this;

		  object IConvertible.ToType(Type conversionType, IFormatProvider provider)
		  {
#if UseDD128
			if (conversionType == typeof(DD128))
					 return (DD128)this;
#endif
			if (conversionType == typeof(object))
					 return this;

				if (Type.GetTypeCode(conversionType) != TypeCode.Object)
					 return Convert.ChangeType(this, Type.GetTypeCode(conversionType), provider);

				throw new InvalidCastException($"Converting type \"{typeof(DD128)}\" to type \"{conversionType.Name}\" is not supported.");
		  }

		  ushort IConvertible.ToUInt16(IFormatProvider provider) => (ushort) this;

		  uint IConvertible.ToUInt32(IFormatProvider provider) => (uint) this;

		  ulong IConvertible.ToUInt64(IFormatProvider provider) => (ulong) this;

		#endregion

		#region Relational Operators

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool operator ==(QD256 left, QD256 right) => left.Equals(right);

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool operator !=(QD256 left, QD256 right) => !left.Equals(right);

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool operator <(QD256 left, QD256 right) => left.CompareTo(right) < 0;

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool operator >(QD256 left, QD256 right) => left.CompareTo(right) > 0;

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool operator <=(QD256 left, QD256 right) => left.CompareTo(right) <= 0;

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public static bool operator >=(QD256 left, QD256 right) => left.CompareTo(right) >= 0;

		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public int CompareTo(QD256 other)
		  {
				var cmp = x0.CompareTo(other.x0);
				if (cmp != 0)
				{
					 return cmp;
				}

				cmp = x1.CompareTo(other.x1);
				if (cmp != 0)
				{
					 return cmp;
				}

				cmp = x2.CompareTo(other.x2);
				if (cmp != 0)
				{
					 return cmp;
				}

				return x3.CompareTo(other.x3);
		  }

		  public int CompareTo(object obj)
		  {
				if (ReferenceEquals(null, obj))
				{
					 return 1;
				}

				return obj is QD256 other
					 ? CompareTo(other)
					 : throw new ArgumentException($"Object must be of type {nameof(QD256)}");
		  }


		  [MethodImpl(MethodImplOptions.AggressiveInlining)]
		  public bool Equals(QD256 other)
				=> x0.Equals(other.x0) &&
					 x1.Equals(other.x1) &&
					 x2.Equals(other.x2) &&
					 x3.Equals(other.x3);

		  public override bool Equals(object obj) => obj is QD256 other && Equals(other);

		  public override int GetHashCode()
		  {
				unchecked
				{
					 var hashCode = x0.GetHashCode();
					 hashCode = (hashCode * 397) ^ x1.GetHashCode();
					 hashCode = (hashCode * 397) ^ x2.GetHashCode();
					 hashCode = (hashCode * 397) ^ x3.GetHashCode();
					 return hashCode;
				}
		  }

		#endregion

		#region Parsing and printing
		public static QD256 Parse(string s, NumberStyles style, IFormatProvider provider)//INumberBase<DdReal>.
		{
			return Parse(s);
		}
		public static QD256 Parse(string s, IFormatProvider provider)
		{
			return Parse(s);
		}
		public static QD256 Parse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider provider)
		{
			return Parse(s.ToString());
		}
		public static QD256 Parse(ReadOnlySpan<char> s, IFormatProvider provider)
		{
			return Parse(s.ToString());
		}

		public static QD256 Parse(string s)
		  {
				if (TryParse(s, out var value))
				{
					 return value;
				}

				throw new FormatException();
		  }
		public static bool TryParse(string s, NumberStyles style, IFormatProvider provider, out QD256 result)
		{
			return TryParse(s, out result);
		}
		public static bool TryParse(string s, IFormatProvider provider, out QD256 result)//IParsable<DdReal>.
		{
			return TryParse(s, out result);
		}
		public static bool TryParse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider provider, out QD256 result)
		{
			return TryParse(s.ToString(), out result);
		}
		public static bool TryParse(ReadOnlySpan<char> s, IFormatProvider provider, out QD256 result)
		{
			return TryParse(s.ToString(), out result);
		}

		public static bool TryParse(string s, out QD256 value)
		{
			return Utils.TryParse(s, out value);
		}

		public static void printComponents(QD256 a, string prefix)
		{
			Debug.WriteLine(string.Format("{0} {1:E} {2:E} {3:E} {4:E}", prefix, a.x0, a.x1, a.x2, a.x3));
		}

		public override string ToString() => ToString("G", CultureInfo.CurrentCulture);

		  public string ToString(IFormatProvider provider)
		  {
				return ToString("G", provider);
		  }

		  public string ToString(string format, IFormatProvider formatProvider)
		  {
				if (formatProvider == null)
				{
					 formatProvider = CultureInfo.CurrentCulture;
				}

			if (string.IsNullOrEmpty(format)) return Utils.ToString(this, digits10 + 1, digits10, false, false, true, MathQD256.ldexp);
			else return x0.ToString(format, formatProvider);
		}
		#endregion

		#region MiscQD
		public void Deconstruct(out double x0, out double x1, out double x2, out double x3)
		{
			x0 = this.x0;
			x1 = this.x1;
			x2 = this.x2;
			x3 = this.x3;
		}

		internal double this[int index]
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				switch (index)
				{
					case 0:
						return x0;
					case 1:
						return x1;
					case 2:
						return x2;
					case 3:
						return x3;
					default:
						throw new IndexOutOfRangeException();
				}
			}
		}
		#endregion

		//*** INumber support

		#region Misc
		public static QD256 operator ^(QD256 a, int b)
		{
			return MathCommon.npwr(a, b);
		}

		public static QD256 operator ^(QD256 a, QD256 b)
		{
			return MathQD256.Exp(b * MathQD256.Log(a));
		}
		static QD256 IDecrementOperators<QD256>.operator --(QD256 value) => value - 1.0d;
		static QD256 IIncrementOperators<QD256>.operator ++(QD256 value) => value + 1.0d;
		static QD256 IUnaryPlusOperators<QD256, QD256>.operator +(QD256 value) => new QD256(+value.x0, +value.x1, +value.x2, +value.x3);
		static QD256 IModulusOperators<QD256, QD256, QD256>.operator %(QD256 left, QD256 right) => (int)left % (int)right;//redo !?
		static QD256 IAdditiveIdentity<QD256, QD256>.AdditiveIdentity => Zero;
		static QD256 IMultiplicativeIdentity<QD256, QD256>.MultiplicativeIdentity => One;
		#endregion

		#region Convert
		//ISpanFormattable
		public bool TryFormat(Span<char> destination, out int charsWritten, [StringSyntax(StringSyntaxAttribute.NumericFormat)] ReadOnlySpan<char> format = default, IFormatProvider provider = null)
		{
			//return Number.TryFormatFloat(m_value, format, NumberFormatInfo.GetInstance(provider), destination, out charsWritten);
			charsWritten = 0;
			return false;
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<QD256>.TryConvertFromChecked<TOther>(TOther value, out QD256 result)
		{
			return TryConvertFrom(value, out result);
		}

		/// <inheritdoc cref="INumberBase{TSelf}.TryConvertFromSaturating{TOther}(TOther, out TSelf)" />
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<QD256>.TryConvertFromSaturating<TOther>(TOther value, out QD256 result)
		{
			return TryConvertFrom(value, out result);
		}

		/// <inheritdoc cref="INumberBase{TSelf}.TryConvertFromTruncating{TOther}(TOther, out TSelf)" />
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<QD256>.TryConvertFromTruncating<TOther>(TOther value, out QD256 result)
		{
			return TryConvertFrom(value, out result);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<QD256>.TryConvertToChecked<TOther>(QD256 value, [MaybeNullWhen(false)] out TOther result)
		{
			// In order to reduce overall code duplication and improve the inlinabilty of these
			// methods for the corelib types we have `ConvertFrom` handle the same sign and
			// `ConvertTo` handle the opposite sign. However, since there is an uneven split
			// between signed and unsigned types, the one that handles unsigned will also
			// handle `Decimal`.
			//
			// That is, `ConvertFrom` for `double` will handle the other signed types and
			// `ConvertTo` will handle the unsigned types

			if (typeof(TOther) == typeof(byte))
			{
				byte actualResult = checked((byte)value);
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(char))
			{
				char actualResult = checked((char)value);
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(decimal))
			{
				decimal actualResult = checked((decimal)value);
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(ushort))
			{
				ushort actualResult = checked((ushort)value);
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(uint))
			{
				uint actualResult = checked((uint)value);
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(ulong))
			{
				ulong actualResult = checked((ulong)value);
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(UInt128))
			{
				UInt128 actualResult = checked((UInt128)(double)value);
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(nuint))
			{
				nuint actualResult = checked((nuint)value);
				result = (TOther)(object)actualResult;
				return true;
			}
			else
			{
				result = default;
				return false;
			}
		}

		/// <inheritdoc cref="INumberBase{TSelf}.TryConvertToSaturating{TOther}(TSelf, out TOther)" />
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<QD256>.TryConvertToSaturating<TOther>(QD256 value, [MaybeNullWhen(false)] out TOther result)
		{
			return TryConvertTo(value, out result);
		}

		/// <inheritdoc cref="INumberBase{TSelf}.TryConvertToTruncating{TOther}(TSelf, out TOther)" />
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<QD256>.TryConvertToTruncating<TOther>(QD256 value, [MaybeNullWhen(false)] out TOther result)
		{
			return TryConvertTo(value, out result);
		}

		private static bool TryConvertTo<TOther>(QD256 value, [MaybeNullWhen(false)] out TOther result)
			 where TOther : INumberBase<TOther>
		{
			// In order to reduce overall code duplication and improve the inlinabilty of these
			// methods for the corelib types we have `ConvertFrom` handle the same sign and
			// `ConvertTo` handle the opposite sign. However, since there is an uneven split
			// between signed and unsigned types, the one that handles unsigned will also
			// handle `Decimal`.
			//
			// That is, `ConvertFrom` for `double` will handle the other signed types and
			// `ConvertTo` will handle the unsigned types

			if (typeof(TOther) == typeof(byte))
			{
				byte actualResult = value >= byte.MaxValue ? byte.MaxValue :
										  value <= byte.MinValue ? byte.MinValue : (byte)value;
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(char))
			{
				char actualResult = value >= char.MaxValue ? char.MaxValue :
										  value <= char.MinValue ? char.MinValue : (char)value;
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(decimal))
			{
				decimal actualResult = value >= +79228162514264337593543950336.0 ? decimal.MaxValue :
											  value <= -79228162514264337593543950336.0 ? decimal.MinValue :
											  IsNaN(value) ? 0.0m : (decimal)value;
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(ushort))
			{
				ushort actualResult = value >= ushort.MaxValue ? ushort.MaxValue :
											 value <= ushort.MinValue ? ushort.MinValue : (ushort)value;
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(uint))
			{
				uint actualResult = value >= uint.MaxValue ? uint.MaxValue :
										  value <= uint.MinValue ? uint.MinValue : (uint)value;
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(ulong))
			{
				ulong actualResult = value >= ulong.MaxValue ? ulong.MaxValue :
											value <= ulong.MinValue ? ulong.MinValue :
											IsNaN(value) ? 0 : (ulong)value;
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(UInt128))
			{
				UInt128 actualResult = value >= 340282366920938463463374607431768211455.0 ? UInt128.MaxValue :
											  value <= 0.0 ? UInt128.MinValue : (UInt128)(double)value;
				result = (TOther)(object)actualResult;
				return true;
			}
			else if (typeof(TOther) == typeof(nuint))
			{
#if TARGET_64BIT
					 nuint actualResult = (value >= ulong.MaxValue) ? unchecked((nuint)ulong.MaxValue) :
												 (value <= ulong.MinValue) ? unchecked((nuint)ulong.MinValue) : (nuint)value;
					 result = (TOther)(object)actualResult;
					 return true;
#else
				nuint actualResult = value >= uint.MaxValue ? uint.MaxValue :
											value <= uint.MinValue ? uint.MinValue : (nuint)value;
				result = (TOther)(object)actualResult;
				return true;
#endif
			}
			else if (typeof(TOther) == typeof(float))
			{
				result = (TOther)(object)(float)value.x0;
				return true;
			}
			else if (typeof(TOther) == typeof(double))
			{
				result = (TOther)(object)value.x0;
				return true;
			}
#if UseDD128
			else if (typeof(TOther) == typeof(DD128))
			{
				result = (TOther)(object)(new DD128(value.x0, value.x1));
				return true;
			}
#endif
			else
			{
				result = default;
				return false;
			}
		}
		private static bool TryConvertFrom<TOther>(TOther value, out QD256 result) where TOther : INumberBase<TOther>
		{
			// In order to reduce overall code duplication and improve the inlinabilty of these
			// methods for the corelib types we have `ConvertFrom` handle the same sign and
			// `ConvertTo` handle the opposite sign. However, since there is an uneven split
			// between signed and unsigned types, the one that handles unsigned will also
			// handle `Decimal`.
			//
			// That is, `ConvertFrom` for `double` will handle the other signed types and
			// `ConvertTo` will handle the unsigned types

			if (typeof(TOther) == typeof(Half))
			{
				Half actualValue = (Half)(object)value;
				result = (double)actualValue;
				return true;
			}
			else if (typeof(TOther) == typeof(short))
			{
				short actualValue = (short)(object)value;
				result = actualValue;
				return true;
			}
			else if (typeof(TOther) == typeof(int))
			{
				int actualValue = (int)(object)value;
				result = actualValue;
				return true;
			}
			else if (typeof(TOther) == typeof(long))
			{
				long actualValue = (long)(object)value;
				result = actualValue;
				return true;
			}
			else if (typeof(TOther) == typeof(Int128))
			{
				Int128 actualValue = (Int128)(object)value;
				result = (double)actualValue;
				return true;
			}
			else if (typeof(TOther) == typeof(nint))
			{
				nint actualValue = (nint)(object)value;
				result = actualValue;
				return true;
			}
			else if (typeof(TOther) == typeof(sbyte))
			{
				sbyte actualValue = (sbyte)(object)value;
				result = actualValue;
				return true;
			}
			else if (typeof(TOther) == typeof(float))
			{
				float actualValue = (float)(object)value;
				result = actualValue;
				return true;
			}
			else if (typeof(TOther) == typeof(double))
			{
				double actualValue = (double)(object)value;
				result = actualValue;
				return true;
			}
			else
			{
				result = default;
				return false;
			}
		}

		#endregion

		#region INumberBase

		static bool INumberBase<QD256>.IsCanonical(QD256 value) => true;

		/// <inheritdoc cref="INumberBase{TSelf}.IsComplexNumber(TSelf)" />
		static bool INumberBase<QD256>.IsComplexNumber(QD256 value) => false;

		/// <inheritdoc cref="INumberBase{TSelf}.IsImaginaryNumber(TSelf)" />
		static bool INumberBase<QD256>.IsImaginaryNumber(QD256 value) => false;
		/// <inheritdoc cref="INumberBase{TSelf}.IsRealNumber(TSelf)" />
		public static bool IsRealNumber(QD256 value)
		{
			// A NaN will never equal itself so this is an
			// easy and efficient way to check for a real number.

#pragma warning disable CS1718
			return value == value;
#pragma warning restore CS1718
		}
		static bool INumberBase<QD256>.IsZero(QD256 value) => IsZero(value);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal static bool IsZero(QD256 d) => d == 0;
		static int INumberBase<QD256>.Radix => 2;

		public static bool IsNegative(QD256 d) => BitConverter.DoubleToInt64Bits(d.x0) < 0;
		public static bool IsPositive(QD256 value) => BitConverter.DoubleToInt64Bits(value.x0) >= 0;
		public static bool IsInteger(QD256 value) => IsFinite(value) && value == Math.Truncate(value.x0);//redo ?!
		public static bool IsEvenInteger(QD256 value) => IsInteger(value) && Math.Abs(value.x0 % 2) == 0;//redo ?!
		public static bool IsOddInteger(QD256 value) => IsInteger(value) && Math.Abs(value.x0 % 2) == 1;//redo ?!
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static QD256 Abs(QD256 a)
		{
			return a.x0 < 0.0 ? -a : a;
		}

		public static bool IsNormal(QD256 d)//todo
		{
			const ulong SignMask = 0x8000_0000_0000_0000;//from MS Double.cs
			const ulong PositiveInfinityBits = 0x7FF0_0000_0000_0000;//from MS Double.cs
			const ulong SmallestNormalBits = 0x0010_0000_0000_0000;//from MS Double.cs

			ulong bits = BitConverter.DoubleToUInt64Bits(d.x0);
			return ((bits & ~SignMask) - SmallestNormalBits) < (PositiveInfinityBits - SmallestNormalBits);
		}

		/// <summary>Determines whether the specified value is subnormal (finite, but not zero or normal).</summary>
		/// <remarks>This effectively checks the value is not NaN, not infinite, not normal, and not zero.</remarks>
		//[NonVersionable]
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsSubnormal(QD256 d)//todo
		{
			const ulong SignMask = 0x8000_0000_0000_0000;//from MS Double.cs
			const ulong MaxTrailingSignificand = 0x000F_FFFF_FFFF_FFFF;//from MS Double.cs

			ulong bits = BitConverter.DoubleToUInt64Bits(d.x0);
			return ((bits & ~SignMask) - 1) < MaxTrailingSignificand;
		}

		//[Intrinsic]
		public static QD256 MaxMagnitude(QD256 x, QD256 y) => Math.MaxMagnitude(x.x0, y.x0);
		//[Intrinsic]
		public static QD256 MaxMagnitudeNumber(QD256 x, QD256 y)
		{
			// This matches the IEEE 754:2019 `maximumMagnitudeNumber` function
			//
			// It does not propagate NaN inputs back to the caller and
			// otherwise returns the input with a larger magnitude.
			// It treats +0 as larger than -0 as per the specification.

			QD256 ax = Abs(x);
			QD256 ay = Abs(y);

			if (ax > ay || IsNaN(ay))
			{
				return x;
			}

			if (ax == ay)
			{
				return IsNegative(x) ? y : x;
			}

			return y;
		}
		//[Intrinsic]
		public static QD256 MinMagnitude(QD256 x, QD256 y) => Math.MinMagnitude(x.x0, y.x0);

		/// <inheritdoc cref="INumberBase{TSelf}.MinMagnitudeNumber(TSelf, TSelf)" />
		//[Intrinsic]
		public static QD256 MinMagnitudeNumber(QD256 x, QD256 y)
		{
			// This matches the IEEE 754:2019 `minimumMagnitudeNumber` function
			//
			// It does not propagate NaN inputs back to the caller and
			// otherwise returns the input with a larger magnitude.
			// It treats +0 as larger than -0 as per the specification.

			QD256 ax = Abs(x);
			QD256 ay = Abs(y);

			if (ax < ay || IsNaN(ay))
			{
				return x;
			}

			if (ax == ay)
			{
				return IsNegative(x) ? x : y;
			}

			return y;
		}
		#endregion

		#region ILogarithmicFunctions
		public static QD256 Log(QD256 x)
		{
			return MathQD256.Log(x);
		}

		public static QD256 Log2(QD256 x)
		{
			return MathQD256.Log2(x);
		}

		public static QD256 Log10(QD256 x)
		{
			return MathQD256.Log10(x);
		}

		public static QD256 Log(QD256 x, QD256 y)
		{
			return MathQD256.Log(x, y);
		}
		#endregion

		#region IFloatingPointConstants
		//
		/// <inheritdoc cref="IFloatingPointConstants{TSelf}.E" />
		static QD256 IFloatingPointConstants<QD256>.E => MathQD256._e;

		/// <inheritdoc cref="IFloatingPointConstants{TSelf}.Pi" />
		static QD256 IFloatingPointConstants<QD256>.Pi => MathQD256._pi;

		/// <inheritdoc cref="IFloatingPointConstants{TSelf}.Tau" />
		static QD256 IFloatingPointConstants<QD256>.Tau => MathQD256._pi * 2;
		#endregion

		#region IPowerFunctions
		/// <inheritdoc cref="IPowerFunctions{TSelf}.Pow(TSelf, TSelf)" />
		//[Intrinsic]
		public static QD256 Pow(QD256 x, QD256 y)
		{
			if (IsZero(x)) return 0;
			if (x < 0 && MathQD256.Floor(y) == y) return MathCommon.npwr(x, (int)y);
			return MathQD256.Exp(y * MathQD256.Log(x));
		}
		#endregion

		#region IExponentialFunctions
		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp" />
		//[Intrinsic]
		public static QD256 Exp(QD256 x) => MathQD256.Exp(x);

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.ExpM1(TSelf)" />
		public static QD256 ExpM1(QD256 x) => MathQD256.Exp(x) - 1;

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp2(TSelf)" />
		public static QD256 Exp2(QD256 x) => Pow(2, x);

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp2M1(TSelf)" />
		public static QD256 Exp2M1(QD256 x) => Pow(2, x) - 1;

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp10(TSelf)" />
		public static QD256 Exp10(QD256 x) => Pow(10, x);

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp10M1(TSelf)" />
		public static QD256 Exp10M1(QD256 x) => Pow(10, x) - 1;
		#endregion

		#region IRootFunctions
		public static QD256 Sqrt(QD256 x)
		{
			return MathQD256.Sqrt(x);
		}

		public static QD256 RootN(QD256 x, int n)
		{
			return MathQD256.RootN(x, n);
		}

		public static QD256 Hypot(QD256 x, QD256 y)
		{
			return Sqrt(MathQD256.Sqr(x) + MathQD256.Sqr(y));
		}

		public static QD256 Cbrt(QD256 x)
		{
			return Pow(x, 1.0 / 3.0);
		}
		#endregion

		#region IMinMaxValue
		//
		/// <inheritdoc cref="IMinMaxValue{TSelf}.MinValue" />
		static QD256 IMinMaxValue<QD256>.MinValue => MinValue;

		/// <inheritdoc cref="IMinMaxValue{TSelf}.MaxValue" />
		static QD256 IMinMaxValue<QD256>.MaxValue => MaxValue;
		#endregion

		#region ITrigonometricFunctions
		//

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Acos(TSelf)" />
		//[Intrinsic]
		public static QD256 Acos(QD256 x) => MathQD256.Acos(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.AcosPi(TSelf)" />
		public static QD256 AcosPi(QD256 x)
		{
			return Acos(x) / Pi;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Asin(TSelf)" />
		//[Intrinsic]
		public static QD256 Asin(QD256 x) => MathQD256.Asin(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.AsinPi(TSelf)" />
		public static QD256 AsinPi(QD256 x)
		{
			return Asin(x) / Pi;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Atan(TSelf)" />
		//[Intrinsic]
		public static QD256 Atan(QD256 x) => MathQD256.Atan(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.AtanPi(TSelf)" />
		public static QD256 AtanPi(QD256 x)
		{
			return Atan(x) / Pi;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Cos(TSelf)" />
		//[Intrinsic]
		public static QD256 Cos(QD256 x) => MathQD256.Cos(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Tan(TSelf)" />
		public static QD256 Tan(QD256 x) => MathQD256.Tan(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.CosPi(TSelf)" />
		public static QD256 CosPi(QD256 x) => MathQD256.CosPi(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.DegreesToRadians(TSelf)" />
		public static QD256 DegreesToRadians(QD256 degrees)
		{
			// NOTE: Don't change the algorithm without consulting the DIM
			// which elaborates on why this implementation was chosen

			return (degrees * Pi) / 180.0;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.RadiansToDegrees(TSelf)" />
		public static QD256 RadiansToDegrees(QD256 radians)
		{
			// NOTE: Don't change the algorithm without consulting the DIM
			// which elaborates on why this implementation was chosen

			return (radians * 180.0) / Pi;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Sin(TSelf)" />
		//[Intrinsic]
		public static QD256 Sin(QD256 x) => MathQD256.Sin(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.SinCos(TSelf)" />
		public static (QD256 Sin, QD256 Cos) SinCos(QD256 x) => MathQD256.SinCos(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.SinCos(TSelf)" />
		public static (QD256 SinPi, QD256 CosPi) SinCosPi(QD256 x) => MathQD256.SinCosPi(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.SinPi(TSelf)" />
		public static QD256 SinPi(QD256 x) => MathQD256.SinPi(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.TanPi(TSelf)" />
		public static QD256 TanPi(QD256 x) => MathQD256.TanPi(x);
		#endregion

		#region IHyperbolicFunctions
		//
		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Acosh(TSelf)" />
		//[Intrinsic]
		public static QD256 Acosh(QD256 x) => MathQD256.Acosh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Asinh(TSelf)" />
		//[Intrinsic]
		public static QD256 Asinh(QD256 x) => MathQD256.Asinh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Atanh(TSelf)" />
		//[Intrinsic]
		public static QD256 Atanh(QD256 x) => MathQD256.Atanh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Cosh(TSelf)" />
		//[Intrinsic]
		public static QD256 Cosh(QD256 x) => MathQD256.Cosh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Sinh(TSelf)" />
		//[Intrinsic]
		public static QD256 Sinh(QD256 x) => MathQD256.Sinh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Tanh(TSelf)" />
		//[Intrinsic]
		public static QD256 Tanh(QD256 x) => MathQD256.Tanh(x);
		#endregion
	 }
}

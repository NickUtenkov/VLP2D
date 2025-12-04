#define UseQD256

using QD256Numeric;
using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using UtilsCommon;
using MathematicCommon;

namespace DD128Numeric
{
	/// <summary>
	///     Represents a floating number with double-double precision (104-bits)
	/// </summary>
	public struct DD128 :
		IConvertible, INumber<DD128>, ITrigonometricFunctions<DD128>, ILogarithmicFunctions<DD128>, IRootFunctions<DD128>,
		IMinMaxValue<DD128>, IPowerFunctions<DD128>, IExponentialFunctions<DD128>, IHyperbolicFunctions<DD128>,
		IFloatingPointConstants<DD128>
		, IAdditionOperators<DD128, double, DD128>, ISubtractionOperators<DD128, double, DD128>
		, IMultiplyOperators<DD128, double, DD128>, IDivisionOperators<DD128, double, DD128>
		, IAdditionOperators<DD128, float, DD128>, ISubtractionOperators<DD128, float, DD128>
		, IMultiplyOperators<DD128, float, DD128>, IDivisionOperators<DD128, float, DD128>
	{
		internal readonly double x0;
		internal readonly double x1;

		public double High => x0;
		public double Low => x1;
		const int digits = 104;
		public const int digits10 = 31;
		public const double PositiveInfinity = (double)1.0 / (double)0.0;
		public static DD128 NaN = MathDD128._nan;
		public static DD128 Zero => MathDD128._zero;
		public static DD128 One => MathDD128._one;
		public static double Epsilon = MathDD128._eps;//1.23259516440783e-32;//= 2^-106
		public static DD128 E = MathDD128._e;
		public static DD128 Pi = MathDD128._pi;//(40 digits are: 3.14159 26535 89793 23846 26433 83279 50288 4197)
		public static DD128 MinValue = -MathDD128._max;
		public static DD128 MaxValue = MathDD128._max;

		public DD128(double x0, double x1 = 0)
		{
			this.x0 = x0;
			this.x1 = x1;
		}

		#region Addition

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator +(DD128 left, DD128 right)
		{
			var (s1, s2) = ArithmeticDD128.TwoSum(left.x0, right.x0);
			var (t1, t2) = ArithmeticDD128.TwoSum(left.x1, right.x1);
			return Renormalize(s1, s2, t1, t2);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator +(DD128 left, double right)
		{
			var (s1, s2) = ArithmeticDD128.TwoSum(left.x0, right);
			return ArithmeticDD128.QuickTwoSum(s1, s2 + left.x1);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator +(double left, DD128 right) => right + left;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator +(DD128 left, float right)
		{
			var (s1, s2) = ArithmeticDD128.TwoSum(left.x0, right);
			return ArithmeticDD128.QuickTwoSum(s1, s2 + left.x1);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator +(float left, DD128 right) => right + left;

		#endregion

		#region Subtraction

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator -(DD128 value) => new DD128(-value.x0, -value.x1);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator -(DD128 left, DD128 right)
		{
			var (s1, s2) = ArithmeticDD128.TwoDiff(left.x0, right.x0);
			var (t1, t2) = ArithmeticDD128.TwoDiff(left.x1, right.x1);
			return Renormalize(s1, s2, t1, t2);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator -(DD128 left, double right)
		{
			var (s1, s2) = ArithmeticDD128.TwoDiff(left.x0, right);
			return ArithmeticDD128.QuickTwoSum(s1, s2 + left.x1);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator -(double left, DD128 right)
		{
			var (s1, s2) = ArithmeticDD128.TwoDiff(left, right.x0);
			return ArithmeticDD128.QuickTwoSum(s1, s2 - right.x1);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator -(DD128 left, float right)
		{
			var (s1, s2) = ArithmeticDD128.TwoDiff(left.x0, right);
			return ArithmeticDD128.QuickTwoSum(s1, s2 + left.x1);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator -(float left, DD128 right)
		{
			var (s1, s2) = ArithmeticDD128.TwoDiff(left, right.x0);
			return ArithmeticDD128.QuickTwoSum(s1, s2 - right.x1);
		}

		#endregion

		#region Multiplication

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator *(DD128 left, DD128 right)
		{
			var (p1, p2) = ArithmeticDD128.TwoProd(left.x0, right.x0);
			return ArithmeticDD128.QuickTwoSum(p1, p2 + left.x0 * right.x1 + left.x1 * right.x0);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator *(DD128 left, double right)
		{
			var (p1, p2) = ArithmeticDD128.TwoProd(left.x0, right);
			return ArithmeticDD128.QuickTwoSum(p1, p2 + left.x1 * right);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator *(double left, DD128 right) => right * left;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator *(DD128 left, float right)
		{
			var (p1, p2) = ArithmeticDD128.TwoProd(left.x0, right);
			return ArithmeticDD128.QuickTwoSum(p1, p2 + left.x1 * right);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator *(float left, DD128 right) => right * left;

		#endregion

		#region Division

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator /(DD128 left, DD128 right)
		{
			var q1 = left.x0 / right.x0;
			if (double.IsInfinity(q1))
			{
				return new DD128(q1);
			}

			var r = left - q1 * right;

			var q2 = r.x0 / right.x0;
			r -= q2 * right;

			var q3 = r.x0 / right.x0;
			return ArithmeticDD128.QuickTwoSum(q1, q2) + q3;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator /(DD128 left, double right)
		{
			var a = left.x0;
			var q1 = a / right;
			if (double.IsInfinity(q1))
			{
				return new DD128(q1);
			}

			var (p1, p2) = ArithmeticDD128.TwoProd(q1, right);
			var (s, e) = ArithmeticDD128.TwoDiff(a, p1);
			e += left.x1;
			e -= p2;

			return ArithmeticDD128.QuickTwoSum(q1, (s + e) / right);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator /(double left, DD128 right) => (DD128)left / right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator /(DD128 left, float right) => left / (DD128)right;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 operator /(float left, DD128 right) => (DD128)left / right;

		#endregion

		#region Other math operators

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsNaN(DD128 dd)
		{
			return double.IsNaN(dd.x0) | double.IsNaN(dd.x1);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsInfinity(DD128 dd)
		{
			return double.IsInfinity(dd.x0);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsFinite(DD128 dd)
		{
			return !(double.IsInfinity(dd.x0) | double.IsNaN(dd.x0));
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsPositiveInfinity(DD128 dd)
		{
			return double.IsPositiveInfinity(dd.x0);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsNegativeInfinity(DD128 dd)
		{
			return double.IsNegativeInfinity(dd.x0);
		}

		#endregion

		#region Conversions

#if UseQD256
		// To DD128
		public static explicit operator DD128(QD256 value) => new DD128(value.x0, value.x1);
#endif
		// TODO: dont throw away the truncated data
		public static explicit operator DD128(decimal value) => new DD128((double)value);

		public static implicit operator DD128(double value) => new DD128(value);

		public static implicit operator DD128(float value) => new DD128(value);

		public static implicit operator DD128(ulong value) => new DD128(value);

		public static implicit operator DD128(long value) => new DD128(value);

		public static implicit operator DD128(uint value) => new DD128(value);

		public static implicit operator DD128(int value) => new DD128(value);

		public static implicit operator DD128(short value) => new DD128(value);

		public static implicit operator DD128(ushort value) => new DD128(value);

		public static implicit operator DD128(byte value) => new DD128(value);

		public static implicit operator DD128(sbyte value) => new DD128(value);

		public static implicit operator DD128(char value) => new DD128(value);

		// From DdReal

		public static explicit operator decimal(DD128 value) => (decimal)value.x0 + (decimal)value.x1;

		public static explicit operator double(DD128 value) => value.x0;//ToDo add value.x ?!

		public static explicit operator float(DD128 value) => (float)value.x0;

		public static explicit operator ulong(DD128 value) => (ulong)value.x0;

		public static explicit operator long(DD128 value) => (long)value.x0;

		public static explicit operator uint(DD128 value) => (uint)value.x0;

		public static explicit operator int(DD128 value) => (int)value.x0;

		public static explicit operator short(DD128 value) => (short)value.x0;

		public static explicit operator ushort(DD128 value) => (ushort)value.x0;

		public static explicit operator byte(DD128 value) => (byte)value.x0;

		public static explicit operator sbyte(DD128 value) => (sbyte)value.x0;

		public static explicit operator char(DD128 value) => (char)value.x0;

		TypeCode IConvertible.GetTypeCode() => TypeCode.Double;

		bool IConvertible.ToBoolean(IFormatProvider provider)
			 => throw new InvalidCastException("Cannot cast DdReal to bool");

		byte IConvertible.ToByte(IFormatProvider provider) => (byte)this;

		char IConvertible.ToChar(IFormatProvider provider) => (char)this;

		DateTime IConvertible.ToDateTime(IFormatProvider provider)
			 => throw new InvalidCastException("Cannot cast DdReal to DateTime");

		decimal IConvertible.ToDecimal(IFormatProvider provider) => (decimal)this;

		double IConvertible.ToDouble(IFormatProvider provider) => (double)this;

		short IConvertible.ToInt16(IFormatProvider provider) => (short)this;

		int IConvertible.ToInt32(IFormatProvider provider) => (int)this;

		long IConvertible.ToInt64(IFormatProvider provider) => (long)this;

		sbyte IConvertible.ToSByte(IFormatProvider provider) => (sbyte)this;

		float IConvertible.ToSingle(IFormatProvider provider) => (float)this;

		object IConvertible.ToType(Type conversionType, IFormatProvider provider)
		{
			if (conversionType == typeof(object)) return this;

			if (Type.GetTypeCode(conversionType) != TypeCode.Object)
				return Convert.ChangeType(this, Type.GetTypeCode(conversionType), provider);

			throw new InvalidCastException($"Converting type \"{typeof(DD128)}\" to type \"{conversionType.Name}\" is not supported.");
		}

		ushort IConvertible.ToUInt16(IFormatProvider provider) => (ushort)this;

		uint IConvertible.ToUInt32(IFormatProvider provider) => (uint)this;

		ulong IConvertible.ToUInt64(IFormatProvider provider) => (ulong)this;

		#endregion

		#region Relational operators

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(DD128 left, DD128 right) => left.Equals(right);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(DD128 left, DD128 right) => !left.Equals(right);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <(DD128 left, DD128 right) => left.CompareTo(right) < 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >(DD128 left, DD128 right) => left.CompareTo(right) > 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <=(DD128 left, DD128 right) => left.CompareTo(right) <= 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >=(DD128 left, DD128 right) => left.CompareTo(right) >= 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public int CompareTo(DD128 other)
		{
			var cmp = x0.CompareTo(other.x0);
			if (cmp != 0)
			{
				return cmp;
			}

			return x1.CompareTo(other.x1);
		}

		public int CompareTo(object obj)
		{
			if (ReferenceEquals(null, obj))
			{
				return 1;
			}

			return obj is DD128 other ? CompareTo(other) : throw new ArgumentException($"Object must be of type {nameof(DD128)}");
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool Equals(DD128 other) => x0.Equals(other.x0) && x1.Equals(other.x1);

		public override bool Equals(object obj) => obj is DD128 other && Equals(other);

		public override int GetHashCode()
		{
			unchecked
			{
				return x0.GetHashCode() * 397 ^ x1.GetHashCode();
			}
		}

		#endregion

		#region Parsing and printing

		public static DD128 Parse(string s)
		{
			if (TryParse(s, out var value)) return value;

			throw new FormatException();
		}

		public static DD128 Parse(string s, NumberStyles style, IFormatProvider provider)
		{
			return Parse(s);
		}
		public static DD128 Parse(string s, IFormatProvider provider)
		{
			return Parse(s);
		}
		public static DD128 Parse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider provider)
		{
			return Parse(s.ToString());
		}
		public static DD128 Parse(ReadOnlySpan<char> s, IFormatProvider provider)
		{
			return Parse(s.ToString());
		}

		public static bool TryParse(string s, out DD128 value)
		{
			return Utils.TryParse(s, out value);
		}

		public static void printComponents(DD128 a, string prefix)
		{
			Debug.WriteLine(string.Format("{0} {1:E} {2:E}", prefix, a.x0, a.x1));
		}

		public static bool TryParse(string s, NumberStyles style, IFormatProvider provider, out DD128 result)
		{
			return TryParse(s, out result);
		}
		public static bool TryParse(string s, IFormatProvider provider, out DD128 result)//IParsable<DdReal>.
		{
			return TryParse(s, out result);
		}
		public static bool TryParse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider provider, out DD128 result)
		{
			return TryParse(s.ToString(), out result);
		}
		public static bool TryParse(ReadOnlySpan<char> s, IFormatProvider provider, out DD128 result)
		{
			return TryParse(s.ToString(), out result);
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

			if (string.IsNullOrEmpty(format)) return Utils.ToString(this, digits10 + 1, digits10, false, false, true, MathDD128.ldexp);
			else return x0.ToString(format, formatProvider);
		}
		#endregion

		#region MiscDD
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void Deconstruct(out double high, out double low)
		{
			high = x0;
			low = x1;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static DD128 Renormalize(double s1, double s2, double t1, double t2)
		{
			s2 += t1;
			DD128 s1s2 = ArithmeticDD128.QuickTwoSum(s1, s2);
			return ArithmeticDD128.QuickTwoSum(s1s2.High, s1s2.Low + t2);
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
					default:
						throw new IndexOutOfRangeException();
				}
			}
		}
		#endregion

		//*** INumber support

		#region Misc
		public static DD128 operator ^(DD128 a, int b)
		{
			return MathCommon.npwr(a, b);
		}

		public static DD128 operator ^(DD128 a, DD128 b)
		{
			return MathDD128.Exp(b * MathDD128.Log(a));
		}

		static DD128 IDecrementOperators<DD128>.operator --(DD128 value) => value - 1.0d;
		static DD128 IIncrementOperators<DD128>.operator ++(DD128 value) => value + 1.0d;
		public static DD128 operator +(DD128 value) => new DD128(+value.x0, +value.x1);
		static DD128 IModulusOperators<DD128, DD128, DD128>.operator %(DD128 left, DD128 right) => (int)left % (int)right;//redo !?
		static DD128 IAdditiveIdentity<DD128, DD128>.AdditiveIdentity => Zero;
		static DD128 IMultiplicativeIdentity<DD128, DD128>.MultiplicativeIdentity => One;
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
		static bool INumberBase<DD128>.TryConvertFromChecked<TOther>(TOther value, out DD128 result)
		{
			return TryConvertFrom(value, out result);
		}

		/// <inheritdoc cref="INumberBase{TSelf}.TryConvertFromSaturating{TOther}(TOther, out TSelf)" />
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<DD128>.TryConvertFromSaturating<TOther>(TOther value, out DD128 result)
		{
			return TryConvertFrom(value, out result);
		}

		/// <inheritdoc cref="INumberBase{TSelf}.TryConvertFromTruncating{TOther}(TOther, out TSelf)" />
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<DD128>.TryConvertFromTruncating<TOther>(TOther value, out DD128 result)
		{
			return TryConvertFrom(value, out result);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<DD128>.TryConvertToChecked<TOther>(DD128 value, [MaybeNullWhen(false)] out TOther result)
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
		static bool INumberBase<DD128>.TryConvertToSaturating<TOther>(DD128 value, [MaybeNullWhen(false)] out TOther result)
		{
			return TryConvertTo(value, out result);
		}

		/// <inheritdoc cref="INumberBase{TSelf}.TryConvertToTruncating{TOther}(TSelf, out TOther)" />
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		static bool INumberBase<DD128>.TryConvertToTruncating<TOther>(DD128 value, [MaybeNullWhen(false)] out TOther result)
		{
			return TryConvertTo(value, out result);
		}

		private static bool TryConvertTo<TOther>(DD128 value, [MaybeNullWhen(false)] out TOther result)
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
				result = (TOther)(object)(float)value.High;
				return true;
			}
			else if (typeof(TOther) == typeof(double))
			{
				result = (TOther)(object)value.High;
				return true;
			}
#if UseQD256
			else if (typeof(TOther) == typeof(QD256))
			{
				result = (TOther)(object)(new QD256(value.High, value.Low));
				return true;
			}
#endif
			else
			{
				result = default;
				return false;
			}
		}

		#endregion

		#region INumberBase
		static bool INumberBase<DD128>.IsCanonical(DD128 value) => true;
		/// <inheritdoc cref="INumberBase{TSelf}.IsComplexNumber(TSelf)" />
		static bool INumberBase<DD128>.IsComplexNumber(DD128 value) => false;

		/// <inheritdoc cref="INumberBase{TSelf}.IsImaginaryNumber(TSelf)" />
		static bool INumberBase<DD128>.IsImaginaryNumber(DD128 value) => false;
		/// <inheritdoc cref="INumberBase{TSelf}.IsRealNumber(TSelf)" />
		public static bool IsRealNumber(DD128 value)
		{
			// A NaN will never equal itself so this is an
			// easy and efficient way to check for a real number.

#pragma warning disable CS1718
			return value == value;
#pragma warning restore CS1718
		}
		static bool INumberBase<DD128>.IsZero(DD128 value) => IsZero(value);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal static bool IsZero(DD128 d) => d == 0;
		static int INumberBase<DD128>.Radix => 2;

		/// <inheritdoc cref="INumberBase{TSelf}.Abs(TSelf)" />
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 Abs(DD128 a)
		{
			return a.High < 0.0 ? -a : a;
		}
		//[Intrinsic]
		public static DD128 MinMagnitude(DD128 x, DD128 y) => Math.MinMagnitude(x.High, y.High);

		/// <inheritdoc cref="INumberBase{TSelf}.MinMagnitudeNumber(TSelf, TSelf)" />
		//[Intrinsic]
		public static DD128 MinMagnitudeNumber(DD128 x, DD128 y)
		{
			// This matches the IEEE 754:2019 `minimumMagnitudeNumber` function
			//
			// It does not propagate NaN inputs back to the caller and
			// otherwise returns the input with a larger magnitude.
			// It treats +0 as larger than -0 as per the specification.

			DD128 ax = Abs(x);
			DD128 ay = Abs(y);

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

		public static bool IsNegative(DD128 d) => BitConverter.DoubleToInt64Bits(d.High) < 0;
		public static bool IsPositive(DD128 value) => BitConverter.DoubleToInt64Bits(value.High) >= 0;

		public static bool IsInteger(DD128 value) => IsFinite(value) && value == Math.Truncate(value.High);
		public static bool IsEvenInteger(DD128 value) => IsInteger(value) && Math.Abs(value.High % 2) == 0;
		public static bool IsOddInteger(DD128 value) => IsInteger(value) && Math.Abs(value.High % 2) == 1;

		public static bool IsNormal(DD128 d)//todo
		{
			const ulong SignMask = 0x8000_0000_0000_0000;//from MS Double.cs
			const ulong PositiveInfinityBits = 0x7FF0_0000_0000_0000;//from MS Double.cs
			const ulong SmallestNormalBits = 0x0010_0000_0000_0000;//from MS Double.cs

			ulong bits = BitConverter.DoubleToUInt64Bits(d.High);
			return ((bits & ~SignMask) - SmallestNormalBits) < (PositiveInfinityBits - SmallestNormalBits);
		}

		/// <summary>Determines whether the specified value is subnormal (finite, but not zero or normal).</summary>
		/// <remarks>This effectively checks the value is not NaN, not infinite, not normal, and not zero.</remarks>
		//[NonVersionable]
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsSubnormal(DD128 d)//todo
		{
			const ulong SignMask = 0x8000_0000_0000_0000;//from MS Double.cs
			const ulong MaxTrailingSignificand = 0x000F_FFFF_FFFF_FFFF;//from MS Double.cs

			ulong bits = BitConverter.DoubleToUInt64Bits(d.High);
			return ((bits & ~SignMask) - 1) < MaxTrailingSignificand;
		}

		//[Intrinsic]
		public static DD128 MaxMagnitude(DD128 x, DD128 y) => Math.MaxMagnitude(x.High, y.High);
		//[Intrinsic]
		public static DD128 MaxMagnitudeNumber(DD128 x, DD128 y)
		{
			// This matches the IEEE 754:2019 `maximumMagnitudeNumber` function
			//
			// It does not propagate NaN inputs back to the caller and
			// otherwise returns the input with a larger magnitude.
			// It treats +0 as larger than -0 as per the specification.

			DD128 ax = Abs(x);
			DD128 ay = Abs(y);

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

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 CreateTruncating<TOther>(TOther value)
#nullable disable
			 where TOther : INumberBase<TOther>
#nullable restore
		{
			DD128 result;

			if (typeof(TOther) == typeof(DD128))
			{
				result = (DD128)(object)value;
			}
			else if (!DD128.TryConvertFrom(value, out result) && !TOther.TryConvertToTruncating<DD128>(value, out result))
			{
				//ThrowHelper.ThrowNotSupportedException();
			}

			return result;
		}

		private static bool TryConvertFrom<TOther>(TOther value, out DD128 result) where TOther : INumberBase<TOther>
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

		#region ITrigonometricFunctions

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Acos(TSelf)" />
		//[Intrinsic]
		public static DD128 Acos(DD128 x) => MathDD128.Acos(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.AcosPi(TSelf)" />
		public static DD128 AcosPi(DD128 x)
		{
			return Acos(x) / Pi;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Asin(TSelf)" />
		//[Intrinsic]
		public static DD128 Asin(DD128 x) => MathDD128.Asin(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.AsinPi(TSelf)" />
		public static DD128 AsinPi(DD128 x)
		{
			return Asin(x) / Pi;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Atan(TSelf)" />
		//[Intrinsic]
		public static DD128 Atan(DD128 x) => MathDD128.Atan(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.AtanPi(TSelf)" />
		public static DD128 AtanPi(DD128 x)
		{
			return Atan(x) / Pi;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Cos(TSelf)" />
		//[Intrinsic]
		public static DD128 Cos(DD128 x) => MathDD128.Cos(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Tan(TSelf)" />
		public static DD128 Tan(DD128 x) => MathDD128.Tan(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.CosPi(TSelf)" />
		public static DD128 CosPi(DD128 x) => MathDD128.CosPi(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.DegreesToRadians(TSelf)" />
		public static DD128 DegreesToRadians(DD128 degrees)
		{
			// NOTE: Don't change the algorithm without consulting the DIM
			// which elaborates on why this implementation was chosen

			return (degrees * Pi) / 180.0;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.RadiansToDegrees(TSelf)" />
		public static DD128 RadiansToDegrees(DD128 radians)
		{
			// NOTE: Don't change the algorithm without consulting the DIM
			// which elaborates on why this implementation was chosen

			return (radians * 180.0) / Pi;
		}

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.Sin(TSelf)" />
		//[Intrinsic]
		public static DD128 Sin(DD128 x) => MathDD128.Sin(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.SinCos(TSelf)" />
		public static (DD128 Sin, DD128 Cos) SinCos(DD128 x) => MathDD128.SinCos(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.SinCosPi(TSelf)" />
		public static (DD128 SinPi, DD128 CosPi) SinCosPi(DD128 x) => MathDD128.SinCosPi(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.SinPi(TSelf)" />
		public static DD128 SinPi(DD128 x) => MathDD128.SinPi(x);

		/// <inheritdoc cref="ITrigonometricFunctions{TSelf}.TanPi(TSelf)" />
		public static DD128 TanPi(DD128 x) => MathDD128.TanPi(x);
		#endregion

		#region ILogarithmicFunctions
		public static DD128 Log(DD128 x)
		{
			return MathDD128.Log(x);
		}

		public static DD128 Log2(DD128 x)
		{
			return MathDD128.Log2(x);
		}

		public static DD128 Log10(DD128 x)
		{
			return MathDD128.Log10(x);
		}

		public static DD128 Log(DD128 x, DD128 y)
		{
			return MathDD128.Log(x, y);
		}
		#endregion

		#region IRootFunctions
		public static DD128 Sqrt(DD128 x)
		{
			return MathDD128.Sqrt(x);
		}

		public static DD128 RootN(DD128 x , int n)
		{
			return MathDD128.RootN(x, n);
		}

		public static DD128 Hypot(DD128 x, DD128 y)
		{
			return Sqrt(MathDD128.Sqr(x) + MathDD128.Sqr(y));
		}

		public static DD128 Cbrt(DD128 x)
		{
			return Pow(x, 1.0 / 3.0);
		}
		#endregion

		#region IMinMaxValue
		/// <inheritdoc cref="IMinMaxValue{TSelf}.MinValue" />
		static DD128 IMinMaxValue<DD128>.MinValue => MinValue;

		/// <inheritdoc cref="IMinMaxValue{TSelf}.MaxValue" />
		static DD128 IMinMaxValue<DD128>.MaxValue => MaxValue;
		#endregion

		#region IFloatingPointConstants
		/// <inheritdoc cref="IFloatingPointConstants{TSelf}.E" />
		static DD128 IFloatingPointConstants<DD128>.E => E;

		/// <inheritdoc cref="IFloatingPointConstants{TSelf}.Pi" />
		static DD128 IFloatingPointConstants<DD128>.Pi => Pi;

		/// <inheritdoc cref="IFloatingPointConstants{TSelf}.Tau" />
		static DD128 IFloatingPointConstants<DD128>.Tau => Pi * 2;
		#endregion

		#region IPowerFunctions
		/// <inheritdoc cref="IPowerFunctions{TSelf}.Pow(TSelf, TSelf)" />
		//[Intrinsic]
		public static DD128 Pow(DD128 x, DD128 y)
		{
			if (IsZero(x)) return 0;
			if (x < 0 && MathDD128.Floor(y) == y) return MathCommon.npwr(x, (int)y);
			return MathDD128.Exp(y * MathDD128.Log(x));
		}
		#endregion

		#region IExponentialFunctions
		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp" />
		//[Intrinsic]
		public static DD128 Exp(DD128 x) => MathDD128.Exp(x);

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.ExpM1(TSelf)" />
		public static DD128 ExpM1(DD128 x) => MathDD128.Exp(x) - 1;

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp2(TSelf)" />
		public static DD128 Exp2(DD128 x) => Pow(2, x);

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp2M1(TSelf)" />
		public static DD128 Exp2M1(DD128 x) => Pow(2, x) - 1;

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp10(TSelf)" />
		public static DD128 Exp10(DD128 x) => Pow(10, x);

		/// <inheritdoc cref="IExponentialFunctions{TSelf}.Exp10M1(TSelf)" />
		public static DD128 Exp10M1(DD128 x) => Pow(10, x) - 1;
		#endregion

		#region IHyperbolicFunctions
		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Acosh(TSelf)" />
		//[Intrinsic]
		public static DD128 Acosh(DD128 x) => MathDD128.Acosh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Asinh(TSelf)" />
		//[Intrinsic]
		public static DD128 Asinh(DD128 x) => MathDD128.Asinh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Atanh(TSelf)" />
		//[Intrinsic]
		public static DD128 Atanh(DD128 x) => MathDD128.Atanh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Cosh(TSelf)" />
		//[Intrinsic]
		public static DD128 Cosh(DD128 x) => MathDD128.Cosh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Sinh(TSelf)" />
		//[Intrinsic]
		public static DD128 Sinh(DD128 x) => MathDD128.Sinh(x);

		/// <inheritdoc cref="IHyperbolicFunctions{TSelf}.Tanh(TSelf)" />
		//[Intrinsic]
		public static DD128 Tanh(DD128 x) => MathDD128.Tanh(x);
		#endregion
	}
}

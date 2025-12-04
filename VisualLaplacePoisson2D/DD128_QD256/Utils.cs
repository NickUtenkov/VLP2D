using MathematicCommon;
using System;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;

namespace UtilsCommon
{
	public static class Utils
	{
		public readonly static char numberDecimalSeparator = CultureInfo.InvariantCulture.NumberFormat.NumberDecimalSeparator[0];

		public static string AppendExpn(string str, int expn)
		{
			int k;

			str += (expn < 0 ? '-' : '+');
			expn = Math.Abs(expn);

			if (expn >= 100)
			{
				k = (expn / 100);
				str += (char)('0' + k);
				expn -= 100 * k;
			}

			k = (expn / 10);
			str += (char)('0' + k);
			expn -= 10 * k;

			return str += (char)('0' + expn);
		}

		public static void RoundString(char[] s, int precision, ref int offset)
		{
			// Input string must be all digits or errors will occur.

			int i;
			int D = precision;

			// Round, handle carry
			if (D > 0 && s[D] >= '5')
			{
				s[D - 1]++;

				i = D - 1;
				while (i > 0 && s[i] > '9')
				{
					s[i] -= (char)10;
					s[--i]++;
				}
			}

			// If first digit is 10, shift everything.
			if (s[0] > '9')
			{
				// e++; // don't modify exponent here
				for (i = precision; i >= 1; i--) s[i + 1] = s[i];
				s[0] = '1';
				s[1] = '0';

				offset++; // now offset needs to be increased by one
				precision++;
			}

			s[precision] = (char)0; // add terminator for array
		}

		public static void Error(string strErr)
		{
			Debug.WriteLine(strErr);
		}

		public static bool TryParse<T>(string s, out T value) where T : INumber<T>, IPowerFunctions<T>
		{
			// TODO: support for IFormatProvider as double.TryParse does

			int p = 0;
			int sign = 0;
			int point = -1;
			int nd = 0;
			int e = 0;
			bool done = false;
			T r = T.Zero;
			T _10 = T.CreateTruncating(10);
			value = r;

			while (!done && p != s.Length)
			{
				char ch = s[p++];
				if (char.IsDigit(ch))
				{
					int d = ch - '0';
					r = r * _10 + T.CreateTruncating(d);
					nd++;
					continue;
				}

				if (ch == numberDecimalSeparator || ch == '.')
				{
					if (point >= 0) return false;
					point = nd;
				}
				else switch (ch)
					{
						case '-':
						case '+':
							if (sign != 0 || nd > 0) return false;
							sign = (ch == '-') ? -1 : 1;
							break;

						case 'E':
						case 'e':
							if (!int.TryParse(s.Substring(p), out e)) return false;
							done = true;
							break;

						default:
							return false;
					}
			}

			if (point >= 0) e -= nd - point;

			if (e != 0) r *= MathCommon.npwr<T>(_10, e);

			value = (sign == -1) ? -r : r;
			return p > 0;
		}

		public static string ToString<T>(T qd, int precision, int digits10, bool isFixed, bool showpos, bool uppercase, Func<T, int, T> ldexp) where T : INumber<T>, ILogarithmicFunctions<T>, IMinMaxValue<T>
		{//dd_real::to_string from dd_real.cpp
			string s = "";
			int i, e = 0;

			if (T.IsNaN(qd)) return uppercase ? "NAN" : "nan";
			if (T.IsPositiveInfinity(qd)) return "+♾";//"INF" : "inf";
			if (T.IsNegativeInfinity(qd)) return "-♾";//∞

			if (qd < T.Zero) s += '-';
			else if (showpos) s += '+';

			if (qd == T.Zero)// Zero case
			{
				s += '0';
				if (precision > 0)
				{
					s += Utils.numberDecimalSeparator;
					s += '0';
				}
			}
			else
			{// Non-zero case
				T abs = T.Abs(qd);
				T l10 = T.Log10(abs);
				T le = T.Log(abs);
				int off = isFixed ? 1 + (int)Math.Floor(double.CreateTruncating(l10)) : 1;
				int d = precision + off;

				int d_with_extra = d;
				if (isFixed) d_with_extra = Math.Max(digits10 + 1, d); // was 60, longer than the max accuracy for DD

				// highly special case - isFixed mode, precision is zero, abs(this) < 1.0
				// without this trap a number like 0.9 printed isFixed with 0 precision prints as 0
				// should be rounded to 1.
				if (isFixed && precision == 0 && T.Abs(qd) < T.One)
				{
					if (T.Abs(qd) >= T.CreateTruncating(0.5)) s += '1';
					else s += '0';

					return s;
				}

				// handle near zero to working precision (but not exactly zero)
				if (isFixed && d <= 0)
				{
					s += '0';
					if (precision > 0)
					{
						s += Utils.numberDecimalSeparator;
					}
				}
				else
				{ // default
					char[] t;
					int j;

					if (isFixed)
					{
						t = new char[d_with_extra + 1];
						ToDigits(qd, t, out e, d_with_extra, ldexp);
					}
					else
					{
						t = new char[d + 1];
						ToDigits(qd, t, out e, d, ldexp);
					}

					off = e + 1;

					if (isFixed)
					{
						// fix the string if it's been computed incorrectly
						// round here in the decimal string if required
						Utils.RoundString(t, d, ref off);

						if (off > 0)
						{
							for (i = 0; i < off; i++) s += t[i];
							if (precision > 0)
							{
								s += Utils.numberDecimalSeparator;
								for (j = 0; j < precision; j++, i++) s += t[i];
							}
						}
						else
						{
							s += "0" + Utils.numberDecimalSeparator;// "0.";
							if (off < 0) s = string.Concat(s, new string('0', -off));
							for (i = 0; i < d; i++) s += t[i];
						}
					}
					else
					{
						s += t[0];
						if (precision > 0) s += Utils.numberDecimalSeparator;

						for (i = 1; i <= precision; i++) s += t[i];

					}
				}
			}

			// trap for improper offset with large values
			// without this trap, output of values of the for 10^j - 1 fail for j > 28
			// and are output with the point in the wrong place, leading to a dramatically off value
			if (isFixed && precision > 0)
			{
				// make sure that the value isn't dramatically larger
				double from_string;
				if (!double.TryParse(s, out from_string))
				{
					if (qd == T.MinValue) return "-1.7976931348623157E+308";
					if (qd == T.MaxValue) return "+1.7976931348623157E+308";
					return "?";
				}

				// if this ratio is large, then we've got problems
				if (Math.Abs(from_string / double.CreateTruncating(qd)) > 3.0)
				{
					// loop on the string, find the point, move it up one
					// don't act on the first character
					for (i = 1; i < s.Length; i++)
					{
						if (s[i] == Utils.numberDecimalSeparator)
						{
							string sub = s.Substring(i - 1, 1);
							s = s.Remove(i, 1).Insert(i, sub);// s[i] = s[i - 1];
							s = s.Remove(i - 1, 1).Insert(i - 1, Utils.numberDecimalSeparator.ToString());// s[i - 1] = '.';
							break;
						}
					}

					from_string = double.Parse(s);
					// if this ratio is large, then the string has not been isFixed
					if (Math.Abs(from_string / double.CreateTruncating(qd)) > 3.0)
					{
						Utils.Error("Re-rounding unsuccessful in large number isFixed point trap.");
					}
				}
			}

			if (!isFixed)
			{
				// Fill in exponent part
				s += uppercase ? 'E' : 'e';
				s = Utils.AppendExpn(s, e);
			}

			return s;
		}

		public static void ToDigits<T>(T qd, char[] s, out int expn, int precision, Func<T, int, T> ldexp) where T : INumber<T>
		{
			int D = precision + 1;  // number of digits to compute

			T r = T.Abs(qd);
			T _10 = T.CreateTruncating(10);
			int e;  // exponent
			int i, d;

			if (double.CreateTruncating(qd) == 0.0)
			{
				expn = 0;
				for (i = 0; i < precision; i++) s[i] = '0';
				return;
			}

			// First determine the (approximate) exponent.
			e = (int)Math.Floor(Math.Log10(Math.Abs(double.CreateTruncating(qd))));

			if (e < -300)
			{
				r *= MathCommon.npwr(_10, 300);
				r /= MathCommon.npwr(_10, (e + 300));
			}
			else if (e > 300)
			{
				r = ldexp(r, -53);
				r /= MathCommon.npwr(_10, e);
				r = ldexp(r, 53);
			}
			else
			{
				r /= MathCommon.npwr(_10, e);
			}

			// Fix exponent if we are off by one
			if (r >= _10)
			{
				r /= _10;
				e++;
			}
			else if (r < T.One)
			{
				r *= _10;
				e--;
			}

			if (r >= _10 || r < T.One)
			{
				Utils.Error("to_digits: can't compute exponent.");
				expn = 0;
				return;
			}

			// Extract the digits
			for (i = 0; i < D; i++)
			{
				d = (int)double.CreateTruncating(r);
				r -= T.CreateTruncating(d);
				r *= _10;

				s[i] = (char)(d + '0');
			}

			// Fix out of range digits.
			for (i = D - 1; i > 0; i--)
			{
				if (s[i] < '0')
				{
					s[i - 1]--;
					s[i] += (char)10;
				}
				else if (s[i] > '9')
				{
					s[i - 1]++;
					s[i] -= (char)10;
				}
			}

			if (s[0] <= '0')
			{
				Utils.Error("to_digits: non-positive leading digit.");
				expn = 0;
				return;
			}

			// Round, handle carry
			if (s[D - 1] >= '5')
			{
				s[D - 2]++;

				i = D - 2;
				while (i > 0 && s[i] > '9')
				{
					s[i] -= (char)10;
					s[--i]++;
				}
			}

			// If first digit is 10, shift everything.
			if (s[0] > '9')
			{
				e++;
				for (i = precision; i >= 2; i--) s[i] = s[i - 1];
				s[0] = '1';
				s[1] = '0';
			}

			s[precision] = (char)0;
			expn = e;
		}
	}
}

using System;
using System.Runtime.CompilerServices;
using UtilsCommon;
using MathematicCommon;

namespace DD128Numeric
{
	internal static class MathDD128
    {
		public static DD128 _2pi = new DD128(6.283185307179586232e+00, 2.449293598294706414e-16);//from dd_const.cpp
		public static DD128 _pi = new DD128(3.141592653589793116e+00, 1.224646799147353207e-16);//from dd_const.cpp(40 digits are: 3.14159 26535 89793 23846 26433 83279 50288 4197)
		public static DD128 _pi2 = new DD128(1.570796326794896558e+00, 6.123233995736766036e-17);//from dd_const.cpp
		public static DD128 _pi16 = new DD128(1.963495408493620697e-01, 7.654042494670957545e-18);//from dd_const.cpp
		public static DD128 _pi4 = new DD128(7.853981633974482790e-01, 3.061616997868383018e-17);//from dd_const.cpp
		public static DD128 _3pi4 = new DD128(2.356194490192344837e+00, 9.1848509936051484375e-17);//from dd_const.cpp

		public static DD128 _e = new DD128(2.718281828459045091e+00, 1.445646891729250158e-16);//from dd_const.cpp
		public static DD128 _log2 = new DD128(6.931471805599452862e-01, 2.319046813846299558e-17);//from dd_const.cpp
		public static DD128 _log10 = new DD128(2.302585092994045901e+00, -2.170756223382249351e-16);//from dd_const.cpp
		public static DD128 _nan = new DD128(double.NaN, double.NaN);
		public static double _eps = 4.93038065763132e-32;  // 2^-104, //from dd_const.cpp
		public static DD128 _max = new DD128(1.79769313486231570815e+308, 9.97920154767359795037e+291);//from dd_const.cpp
		public static DD128 _zero => new DD128(0);
		public static DD128 _one => new DD128(1);

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 Sqr(double a)
		{
			double p1, p2;
			p1 = ArithmeticDD128.TwoSqr(a, out p2);
			return new DD128(p1, p2);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 Sqr(DD128 a)
		{
			double p1, p2;
			p1 = ArithmeticDD128.TwoSqr(a.High, out p2);
			p2 += 2.0 * a.High * a.Low;
			p2 += a.Low * a.Low;
			return ArithmeticDD128.QuickTwoSum(p1, p2);
		}

		public static DD128 Sqrt(DD128 a)
		{
			/* Strategy:  Use Karp's trick:  if x is an approximation
				to sqrt(a), then

					sqrt(a) = a*x + [a - (a*x)^2] * x / 2   (approx)

				The approximation is accurate to twice the accuracy of x.
				Also, the multiplication (a*x) and [-]*x can be done with
				only half the precision.
			*/

			if (DD128.IsZero(a)) return 0.0;

			if (DD128.IsNegative(a))
			{
				Utils.Error("ERROR (DD128::sqrt): Negative argument.");
				return 0.0;
			}

			double x = 1.0 / Math.Sqrt(a.High);
			double ax = a.High * x;
			return ArithmeticDD128.TwoSum(ax, (a - Sqr(ax)).High * (x * 0.5));
		}
		/* Computes the n-th root of the double-double number a.
			NOTE: n must be a positive integer.  
			NOTE: If n is even, then a must not be negative.       */
		public static DD128 RootN(DD128 a, int n)
		{
			/* Strategy:  Use Newton iteration for the function

					  f(x) = x^(-n) - a

				to find its root a^{-1/n}.  The iteration is thus

					  x' = x + x * (1 - a * x^n) / n

				which converges quadratically.  We can then find 
			  a^{1/n} by taking the reciprocal.
			*/

			if (n <= 0)
			{
				Utils.Error("(DD128::nroot): N must be positive.");
				return _nan;
			}

			if (n % 2 == 0 && DD128.IsNegative(a))
			{
				Utils.Error("(DD128::nroot): Negative argument.");
				return _nan;
			}

			if (n == 1)
			{
				return a;
			}
			if (n == 2)
			{
				return Sqrt(a);
			}

			if (DD128.IsZero(a))
				return 0.0;

			/* Note  a^{-1/n} = exp(-log(a)/n) */
			DD128 r = DD128.Abs(a);
			DD128 x = Math.Exp(-Math.Log(r.x0) / n);

			/* Perform Newton's iteration. */
			x += x * (1.0 - r * MathCommon.npwr(x, n)) / (double)(n);
			if (a.x0 < 0.0)
				x = -x;
			return 1.0 / x;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 Floor(DD128 a)
		{
			double hi = Math.Floor(a.High);
			double lo = 0.0;

			if (hi == a.High)
			{
				// High word is integer already.  Round the low word.
				lo = Math.Floor(a.Low);
				(hi, lo) = ArithmeticDD128.QuickTwoSum(hi, lo);
			}

			return new DD128(hi, lo);
		}
		// Table of sin(k * pi/16) and cos(k * pi/16).
		static double[] sin_table = new double[]{
		  1.950903220161282758e-01, -7.991079068461731263e-18,
		  3.826834323650897818e-01, -1.005077269646158761e-17,
		  5.555702330196021776e-01,  4.709410940561676821e-17,
		  7.071067811865475727e-01, -4.833646656726456726e-17
		};

		static double[] cos_table = new double[] {
		  9.807852804032304306e-01, 1.854693999782500573e-17,
		  9.238795325112867385e-01, 1.764504708433667706e-17,
		  8.314696123025452357e-01, 1.407385698472802389e-18,
		  7.071067811865475727e-01, -4.833646656726456726e-17
		};
		public static DD128 Sin(DD128 a)
		{
			/* Strategy.  To compute sin(x), we choose integers a, b so that

				  x = s + a * (pi/2) + b * (pi/16)

				and |s| <= pi/32.  Using the fact that 

				  sin(pi/16) = 0.5 * sqrt(2 - sqrt(2 + sqrt(2)))

				we can compute sin(x) from sin(s), cos(s).  This greatly 
				increases the convergence of the sine Taylor series. */

			if (DD128.IsZero(a)) return 0.0;

			// approximately reduce modulo 2*pi
			DD128 z = Floor(a / _2pi + 0.5);
			DD128 r = a - _2pi * z;

			// approximately reduce modulo pi/2 and then modulo pi/16.
			DD128 t;
			double q = Math.Floor(r.High / _pi2.High + 0.5);
			t = r - _pi2 * q;
			int j = (int)(q);
			q = Math.Floor(t.High / _pi16.High + 0.5);
			t -= _pi16 * q;
			int k = (int)(q);
			int abs_k = Math.Abs(k);

			if (j < -2 || j > 2)
			{
				Utils.Error("(DD128::sin): Cannot reduce modulo pi/2.");
				return _nan;
			}

			if (abs_k > 4)
			{
				Utils.Error("(DD128::sin): Cannot reduce modulo pi/16.");
				return _nan;
			}

			if (k == 0)
			{
				switch (j)
				{
					case 0:
						return sin_taylor(t);
					case 1:
						return cos_taylor(t);
					case -1:
						return -cos_taylor(t);
					default:
						return -sin_taylor(t);
				}
			}

			DD128 u = new DD128 (cos_table[(abs_k - 1) * 2 + 0], cos_table[(abs_k - 1) * 2 + 1]);
			DD128 v = new DD128(sin_table[(abs_k - 1) * 2 + 0], sin_table[(abs_k - 1) * 2 + 1]);
			DD128 sin_t, cos_t;
			sincos_taylor(t, out sin_t, out cos_t);
			if (j == 0)
			{
				if (k > 0) r = u * sin_t + v * cos_t;
				else r = u * sin_t - v * cos_t;
			}
			else if (j == 1)
			{
				if (k > 0) r = u * cos_t - v * sin_t;
				else r = u * cos_t + v * sin_t;
			}
			else if (j == -1)
			{
				if (k > 0) r = v * sin_t - u * cos_t;
				else if (k < 0) r = -u * cos_t - v * sin_t;
			}
			else
			{
				if (k > 0) r = -u * sin_t - v * cos_t;
				else r = v * cos_t - u * sin_t;
			}

			return r;
		}

		public static DD128 Cos(DD128 a)
		{
			if (DD128.IsZero(a)) return 1.0;

			// approximately reduce modulo 2*pi
			DD128 z = Floor(a / _2pi + 0.5);
			DD128 r = a - z * _2pi;

			// approximately reduce modulo pi/2 and then modulo pi/16
			DD128 t;
			double q = Math.Floor(r.High / _pi2.High + 0.5);
			t = r - _pi2 * q;
			int j = (int)(q);
			q = Math.Floor(t.High / _pi16.High + 0.5);
			t -= _pi16 * q;
			int k = (int)(q);
			int abs_k = Math.Abs(k);

			if (j < -2 || j > 2)
			{
				Utils.Error("(DD128.Cos): Cannot reduce modulo pi/2.");
				return _nan;
			}

			if (abs_k > 4)
			{
				Utils.Error("(DD128.Cos): Cannot reduce modulo pi/16.");
				return _nan;
			}

			if (k == 0)
			{
				switch (j)
				{
					case 0:
						return cos_taylor(t);
					case 1:
						return -sin_taylor(t);
					case -1:
						return sin_taylor(t);
					default:
						return -cos_taylor(t);
				}
			}

			DD128 sin_t, cos_t;
			sincos_taylor(t, out sin_t, out cos_t);
			DD128 u = new DD128(cos_table[(abs_k - 1) * 2 + 0], cos_table[(abs_k - 1) * 2 + 1]);
			DD128 v = new DD128(sin_table[(abs_k - 1) * 2 + 0], sin_table[(abs_k - 1) * 2 + 1]);

			if (j == 0)
			{
				if (k > 0) r = u * cos_t - v * sin_t;
				else r = u * cos_t + v * sin_t;
			}
			else if (j == 1)
			{
				if (k > 0) r = -u * sin_t - v * cos_t;
				else r = v * cos_t - u * sin_t;
			}
			else if (j == -1)
			{
				if (k > 0) r = u * sin_t + v * cos_t;
				else r = u * sin_t - v * cos_t;
			}
			else
			{
				if (k > 0) r = v * sin_t - u * cos_t;
				else r = -u * cos_t - v * sin_t;
			}

			return r;
		}

		public static DD128 Tan(DD128 a)
		{
			DD128 s, c;
			sincos(a, out s, out c);
			return s / c;
		}

		public static (DD128, DD128) SinCos(DD128 a)
		{
			DD128 s, c;
			sincos(a, out s, out c);
			return (s, c);
		}

		public static DD128 SinPi(DD128 a)
		{
			return Sin(a * _pi);//for smart realization see MS Double.cs
		}

		public static DD128 CosPi(DD128 a)
		{
			return Cos(a * _pi);//for smart realization see MS Double.cs
		}

		public static DD128 TanPi(DD128 a)
		{
			return Cos(a * _pi);//for smart realization see MS Double.cs
		}

		public static (DD128, DD128) SinCosPi(DD128 a)
		{
			DD128 s, c;
			sincos(a * _pi, out s, out c);//for smart realization see MS Double.cs
			return (s, c);
		}

		static DD128 sin_taylor(DD128 a)
		{
			double thresh = 0.5 * Math.Abs((double)a) * _eps;
			DD128 r, s, t, x;

			if (DD128.IsZero(a))
			{
				return 0.0;
			}

			int i = 0;
			x = -Sqr(a);
			s = a;
			r = a;
			do
			{
				r *= x;
				t = r * new DD128(inv_fact[i,0], inv_fact[i,1]);
				s += t;
				i += 2;
			} while (i < n_inv_fact && Math.Abs((double)t) > thresh);

			return s;
		}

		static DD128 cos_taylor(DD128 a)
		{
			double thresh = 0.5 * _eps;
			DD128 r, s, t, x;

			if (DD128.IsZero(a))
			{
				return 1.0;
			}

			x = -Sqr(a);
			r = x;
			s = 1.0 + mul_pwr2(r, 0.5);
			int i = 1;
			do
			{
				r *= x;
				t = r * new DD128(inv_fact[i,0], inv_fact[i,1]);
				s += t;
				i += 2;
			} while (i < n_inv_fact && Math.Abs((double)t) > thresh);

			return s;
		}

		static void sincos_taylor(DD128 a, out DD128 sin_a, out DD128 cos_a)
		{
			if (DD128.IsZero(a))
			{
				sin_a = 0.0;
				cos_a = 1.0;
				return;
			}

			sin_a = sin_taylor(a);
			cos_a = Sqrt(1.0 - Sqr(sin_a));
		}

		static void sincos(DD128 a, out DD128 sin_a, out DD128 cos_a)
		{
			if (DD128.IsZero(a))
			{
				sin_a = 0.0;
				cos_a = 1.0;
				return;
			}

			// approximately reduce modulo 2*pi
			DD128 z = nint(a / _2pi);
			DD128 r = a - _2pi * z;

			// approximately reduce module pi/2 and pi/16
			DD128 t;
			double q = Math.Floor(r.x0 / _pi2.x0 + 0.5);
			t = r - _pi2 * q;
			int j = (int)(q);
			int abs_j = Math.Abs(j);
			q = Math.Floor(t.x0 / _pi16.x0 + 0.5);
			t -= _pi16 * q;
			int k = (int)(q);
			int abs_k = Math.Abs(k);

			if (abs_j > 2)
			{
				Utils.Error("(DD128::sincos): Cannot reduce modulo pi/2.");
				cos_a = sin_a = _nan;
				return;
			}

			if (abs_k > 4)
			{
				Utils.Error("(DD128::sincos): Cannot reduce modulo pi/16.");
				cos_a = sin_a = _nan;
				return;
			}

			DD128 sin_t, cos_t;
			DD128 s, c;

			sincos_taylor(t, out sin_t, out cos_t);

			if (abs_k == 0)
			{
				s = sin_t;
				c = cos_t;
			}
			else
			{
				DD128 u = new DD128(cos_table[(abs_k - 1) * 2 + 0], cos_table[(abs_k - 1) * 2 + 1]);
				DD128 v = new DD128(sin_table[(abs_k - 1) * 2 + 0], sin_table[(abs_k - 1) * 2 + 1]);

				if (k > 0)
				{
					s = u * sin_t + v * cos_t;
					c = u * cos_t - v * sin_t;
				}
				else
				{
					s = u * sin_t - v * cos_t;
					c = u * cos_t + v * sin_t;
				}
			}

			if (abs_j == 0)
			{
				sin_a = s;
				cos_a = c;
			}
			else if (j == 1)
			{
				sin_a = c;
				cos_a = -s;
			}
			else if (j == -1)
			{
				sin_a = -c;
				cos_a = s;
			}
			else
			{
				sin_a = -s;
				cos_a = -c;
			}
		}

		/* Round to Nearest integer */
		static DD128 nint(DD128 a)
		{
			double hi = MathCommon.nint(a.x0);
			double lo;

			if (hi == a.x0)
			{
				/* High word is an integer already.  Round the low word.*/
				lo = MathCommon.nint(a.x1);

				/* Renormalize. This is needed if x[0] = some integer, x[1] = 1/2.*/
				hi = ArithmeticDD128.QuickTwoSum(hi, lo, out lo);
			}
			else
			{
				/* High word is not an integer. */
				lo = 0.0;
				if (Math.Abs(hi - a.x0) == 0.5 && a.x1 < 0.0)
				{
					/* There is a tie in the high word, consult the low word to break the tie. */
					hi -= 1.0;      /* NOTE: This does not cause INEXACT. */
				}
			}

			return new DD128(hi, lo);
		}

		public static DD128 Asin(DD128 a)
		{
			DD128 abs_a = DD128.Abs(a);

			if (abs_a > 1.0)
			{
				Utils.Error("ERROR DD128.asin: Argument out of domain.");
				return 0.0;
			}

			if (abs_a == _one) return (DD128.IsPositive(a)) ? _pi2 : -_pi2;

			return Atan2(a, Sqrt(1.0 - Sqr(a)));
		}

		public static DD128 Acos(DD128 a)
		{
			DD128 abs_a = DD128.Abs(a);

			if (abs_a > 1.0)
			{
				Utils.Error("ERROR DD128.acos: Argument out of domain.");
				return 0.0;
			}

			if (abs_a == _one) return (DD128.IsPositive(a)) ? _zero : _pi;

			return Atan2(Sqrt(1.0 - Sqr(a)), a);
		}

		public static DD128 Atan(DD128 a)
		{
			return Atan2(a, _one);
		}

		public static DD128 Atan2(DD128 y, DD128 x)
		{
			/* Strategy: Instead of using Taylor series to compute 
				arctan, we instead use Newton's iteration to solve
				the equation

					sin(z) = y/r    or    cos(z) = x/r

				where r = sqrt(x^2 + y^2).
				The iteration is given by

					z' = z + (y - sin(z)) / cos(z)          (for equation 1)
					z' = z - (x - cos(z)) / sin(z)          (for equation 2)

				Here, x and y are normalized so that x^2 + y^2 = 1.
				If |x| > |y|, then first iteration is used since the 
				denominator is larger.  Otherwise, the second is used.
			*/

			if (DD128.IsZero(x))
			{
				if (DD128.IsZero(y))
				{
					// Both x and y is zero.
					Utils.Error("ERROR DD128.atan2: Both arguments zero.");
					return 0.0;
				}

				return (DD128.IsPositive(y)) ? _pi2 : -_pi2;
			}
			else if (DD128.IsZero(y))
			{
				return (DD128.IsPositive(x)) ? _zero : _pi;
			}

			if (x == y)
			{
				return (DD128.IsPositive(y)) ? _pi4 : -_3pi4;
			}

			if (x == -y)
			{
				return (DD128.IsPositive(y)) ? _3pi4 : -_pi4;
			}

			DD128 r = Sqrt(Sqr(x) + Sqr(y));
			DD128 xx = x / r;
			DD128 yy = y / r;

			// Compute double precision approximation to atan.
			DD128 z = double.Atan2((double)y, (double)x);
			DD128 sin_z, cos_z;

			if (Math.Abs(xx.x0) > Math.Abs(yy.x0))
			{
				// Use Newton iteration 1.  z' = z + (y - sin(z)) / cos(z)
				sincos(z, out sin_z, out cos_z);
				z += (yy - sin_z) / cos_z;
			}
			else
			{
				// Use Newton iteration 2.  z' = z - (x - cos(z)) / sin(z)
				sincos(z, out sin_z, out cos_z);
				z -= (xx - cos_z) / sin_z;
			}

			return z;
		}

		/* Logarithm.  Computes log(x) in double-double precision.
			This is a natural logarithm (i.e., base e).            */
		public static DD128 Log(DD128 a)
		{
			/* Strategy.  The Taylor series for log converges much more
				slowly than that of exp, due to the lack of the factorial
				term in the denominator.  Hence this routine instead tries
				to determine the root of the function

					 f(x) = exp(x) - a

				using Newton iteration.  The iteration is given by

					 x' = x - f(x)/f'(x) 
						 = x - (1 - a * exp(-x))
						 = x + a * exp(-x) - 1.

				Only one iteration is needed, since Newton's iteration
				approximately doubles the number of digits per iteration. */

			if (a == _one) return 0.0;

			if (a.High <= 0.0)
			{
				Utils.Error("(DD128::log): Non-positive argument.");
				return _nan;
			}

			DD128 x = Math.Log(a.High);   /* Initial approximation */

			DD128 res = a * Exp(-x);
			if (!DD128.IsNaN(res)) x += res;// x = x + a * Exp(-x) - 1.0;
			x -= 1;
			//x = x + a * Exp(-x) - 1.0;
			return x;
		}

		public static DD128 Log2(DD128 a)
		{
			return Log(a) / _log2;
		}

		public static DD128 Log10(DD128 a)
		{
			return Log(a) / _log10;
		}

		public static DD128 Log(DD128 a, DD128 b)
		{
			return Log(a) / Log(b);
		}

		const int n_inv_fact = 15;
		static double[,] inv_fact = new double[n_inv_fact, 2]
		{
		  { 1.66666666666666657e-01,  9.25185853854297066e-18},
		  { 4.16666666666666644e-02,  2.31296463463574266e-18},
		  { 8.33333333333333322e-03,  1.15648231731787138e-19},
		  { 1.38888888888888894e-03, -5.30054395437357706e-20},
		  { 1.98412698412698413e-04,  1.72095582934207053e-22},
		  { 2.48015873015873016e-05,  2.15119478667758816e-23},
		  { 2.75573192239858925e-06, -1.85839327404647208e-22},
		  { 2.75573192239858883e-07,  2.37677146222502973e-23},
		  { 2.50521083854417202e-08, -1.44881407093591197e-24},
		  { 2.08767569878681002e-09, -1.20734505911325997e-25},
		  { 1.60590438368216133e-10,  1.25852945887520981e-26},
		  { 1.14707455977297245e-11,  2.06555127528307454e-28},
		  { 7.64716373181981641e-13,  7.03872877733453001e-30},
		  { 4.77947733238738525e-14,  4.39920548583408126e-31},
		  { 2.81145725434552060e-15,  1.65088427308614326e-31}
		};

		/* Exponential.  Computes exp(x) in double-double precision. */
		public static DD128 Exp(DD128 a)
		{
			/* Strategy:  We first reduce the size of x by noting that

					  exp(kr + m * log(2)) = 2^m * exp(r)^k

				where m and k are integers.  By choosing m appropriately
				we can make |kr| <= log(2) / 2 = 0.347.  Then exp(r) is 
				evaluated using the familiar Taylor series.  Reducing the 
				argument substantially speeds up the convergence.       */

			const int k = 64;

			if (a.High <= -709.0) return 0.0;

			if (a.High >= 709.0) return DD128.PositiveInfinity;

			if (a == _zero) return 1.0;

			if (a == _one) return _e;
			int z = (int)Math.Floor((a / _log2).High);
			DD128 r = (a - _log2 * (double)z) / (double)k;
			DD128 s, t, f, p;
			double m;

			s = 1.0 + r;
			p = Sqr(r);
			m = 2.0;
			f = 2.0;
			t = p / f;
			do
			{
				s += t;
				p *= r;
				m += 1.0;
				f *= m;
				t = p / f;
			} while (DD128.Abs(t) > 1.0e-35);

			s += t;
			r = MathCommon.npwr(s, k);
			r *= Math.Pow(2.0, z);

			return r;
		}

		public static DD128 ldexp(DD128 a, int n)
		{
			return new DD128(ldexp(a.x0, n), ldexp(a.x1, n));
		}

		static double ldexp(double a, int n)
		{
			return Math.ScaleB(a, n);//a * Math.Pow(2.0, n);
		}

		// Computes  dd * d  where d is known to be a power of 2
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static DD128 mul_pwr2(DD128 dd, double b)
		{
			return new DD128(dd.High * b, dd.Low * b);
		}

		static DD128 inv(DD128 a)
		{
			return 1.0 / a;
		}

		public static DD128 Sinh(DD128 a)
		{
			if (DD128.IsZero(a))
			{
				return 0.0;
			}

			if (DD128.Abs(a) > 0.05)
			{
				DD128 ea = Exp(a);
				return mul_pwr2(ea - inv(ea), 0.5);
			}

			/* since a is small, using the above formula gives
				a lot of cancellation.  So use Taylor series.   */
			DD128 s = a;
			DD128 t = a;
			DD128 r = Sqr(t);
			double m = 1.0;
			double thresh = Math.Abs(a.x0 * _eps);

			do
			{
				m += 2.0;
				t *= r;
				t /= (m - 1) * m;

				s += t;
			} while (DD128.Abs(t) > thresh);

			return s;

		}

		public static DD128 Cosh(DD128 a)
		{
			if (DD128.IsZero(a))
			{
				return 1.0;
			}

			DD128 ea = Exp(a);
			return mul_pwr2(ea + inv(ea), 0.5);
		}

		public static DD128 Tanh(DD128 a)
		{
			if (DD128.IsZero(a))
			{
				return 0.0;
			}

			if (Math.Abs(a.x0) > 0.05)
			{
				DD128 ea = Exp(a);
				DD128 inv_ea = inv(ea);
				return (ea - inv_ea) / (ea + inv_ea);
			}
			else
			{
				DD128 s, c;
				s = Sinh(a);
				c = Sqrt(1.0 + Sqr(s));
				return s / c;
			}
		}

		public static DD128 Asinh(DD128 a)
		{
			return Log(a + Sqrt(Sqr(a) + 1.0));
		}

		public static DD128 Acosh(DD128 a)
		{
			if (a < 1.0)
			{
				Utils.Error("(DD128::acosh): Argument out of domain.");
				return _nan;
			}

			return Log(a + Sqrt(Sqr(a) - 1.0));
		}

		public static DD128 Atanh(DD128 a)
		{
			if (DD128.Abs(a) >= 1.0)
			{
				Utils.Error("(DD128::atanh): Argument out of domain.");
				return _nan;
			}

			return mul_pwr2(Log((1.0 + a) / (1.0 - a)), 0.5);
		}
	}
}

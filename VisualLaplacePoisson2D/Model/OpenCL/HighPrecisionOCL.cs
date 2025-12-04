namespace VLP2D.Common
{
	public static class HighPrecisionOCL
	{
		public static string strSingleDefines =
@"
#define HP(a)	(a)
#define Zero	0.0f
#define One		1.0f
#define Pi4    0.78539816f
#define sqr(x) (x)*(x)
#define to_double(x) (x)
#define to_float(x) ((float)(x))

";
		public static string strDoubleDefines =
@"
#define HP(a)	(a)
#define Zero	0.0
#define One		1.0
#define Pi4    0.78539816339744830961566084581988//from Windows Calculator
#define sqr(x) (x)*(x)
#define to_double(x) (x)
#define to_float(x) ((float)(x))

";
		public static string strHighPrecision_Basic =
@"
#define Use_SPLIT_THRESH
//from inline.cu
#define _GQD_SPLITTER            (134217729.0)                   // = 2^27 + 1
#define _GQD_SPLIT_THRESH        (6.69692879491417e+299)         // = 2^996

// Basic Funcitons

//computs fl( a + b ) and err( a + b ), assumes |a| > |b|

double quick_two_sum(double a, double b, double *err)
{
	//not use(we are NOT on CPU)
	/*if (b == 0.0)
	{
		  *err = 0.0;
		  return (a + b);
	 }*/

	 double s = a + b;
	 *err = b - (s - a);

	 return s;
}

double two_sum(double a, double b, double *err)
{
	//not use(we are NOT on CPU)
	//nnn why should use ?!
	if ((a == 0.0) || (b == 0.0))
	{
		*err = 0.0;
		return (a + b);
	}

	double s = a + b;
	 double bb = s - a;
	 *err = (a - (s - bb)) + (b - bb);

	 return s;
}


//computes fl( a - b ) and err( a - b ), assumes |a| >= |b|

double quick_two_diff(double a, double b, double *err)
{
	//not use(we are NOT on CPU)
	/*if (a == b)
	{
		  *err = 0.0;
		  return 0.0;
	 }*/

	 double s;

	 s = a - b;
	 *err = (a - s) - b;
	 return s;
}

//computes fl( a - b ) and err( a - b )

double two_diff(double a, double b, double *err)
{
	//not use(we are NOT on CPU)
	/*if (a == b)
	{
		  *err = 0.0;
		  return 0.0;
	 }*/

	 double s = a - b;

	 double bb = s - a;
	 *err = (a - (s - bb)) - (b + bb);
	 return s;
}

// Computes high word and lo word of a 

void split(double a, double *hi, double *lo)
{
	double temp;
	#ifdef Use_SPLIT_THRESH
	if (a > _GQD_SPLIT_THRESH || a < -_GQD_SPLIT_THRESH)
	{
		a *= 3.7252902984619140625e-09; // 2^-28
		temp = _GQD_SPLITTER * a;
		*hi = temp - (temp - a);
		*lo = a - *hi;
		*hi *= 268435456.0; // 2^28
		*lo *= 268435456.0; // 2^28
	}
	else
	#endif
	{
		temp = _GQD_SPLITTER * a;
		*hi = temp - (temp - a);
		*lo = a - *hi;
	}
}

/* Computes fl(a*b) and err(a*b). */
double two_prod(double a, double b, double *err) {

	 double a_hi, a_lo, b_hi, b_lo;
	 double p = a * b;
	 split(a, &a_hi, &a_lo);
	 split(b, &b_hi, &b_lo);

	 *err = (a_hi * b_hi) - p + (a_hi * b_lo) + (a_lo * b_hi) + (a_lo * b_lo);

	 return p;
}

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
double two_sqr(double a, double *err) {
	 double hi, lo;
	 double q = a * a;
	 split(a, &hi, &lo);
	 *err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
	 return q;
}

/* Computes the nearest integer to d. */
__attribute__((overloadable))
double nint(double d) {
	 if (d == floor(d))
		  return d;
	 return floor(d + 0.5);
}

double __dmul_rn(double a, double b)
{//CUDA Multiply two floating-point values in round-to-nearest-even mode.
	return a * b;
}

double __dadd_rn(double a, double b)
{//CUDA Add two floating-point values in round-to-nearest-even mode.
	return a + b;
}

";
		public static string strTypeDefDD128 =
@"
//#define UseDouble2
#ifdef UseDouble2
typedef double2 gdd_real;
#else
typedef struct __attribute__ ((packed)) dbl2
{
	double x,y;
} gdd_real;
#endif
typedef gdd_real DD128;
";
		public static string strDD128 = strTypeDefDD128 +
@"
//from gdd_basic.cu
#define Zero	make_dd(0.0)
#define One		make_dd(1.0)

// type construction
__attribute__((overloadable))
__attribute__((always_inline))
gdd_real make_dd(const double x, const double y)
{
	#ifdef UseDouble2
	return (gdd_real)(x, y);
	#else
	return (gdd_real){x, y};
	#endif
}

__attribute__((overloadable))
__attribute__((always_inline))
gdd_real make_dd(const double x)
{
	#ifdef UseDouble2
	return (gdd_real)(x, 0.0);
	#else
	return (gdd_real){x, 0.0};
	#endif
}
///////////////////// Addition /////////////////////

gdd_real positive(gdd_real a) {
	 return make_dd(fabs(a.x), fabs(a.y));
}

gdd_real negative(gdd_real a) {
	 return make_dd(-a.x, -a.y);
}

// double-double = double + double
gdd_real dd_add(double a, double b) {
	 double s, e;
	 s = two_sum(a, b, &e);
	 return make_dd(s, e);
}

// double-double + double
gdd_real add_HD(gdd_real a, double b) {
	 double s1, s2;
	 s1 = two_sum(a.x, b, &s2);
	 s2 += a.y;
	 s1 = quick_two_sum(s1, s2, &s2);
	 return make_dd(s1, s2);
}

// double + double-double
gdd_real add_DH(double a, gdd_real b) {
	 return add_HD(b, a);
}

gdd_real sloppy_add(gdd_real a, gdd_real b) {
	 double s, e;

	 s = two_sum(a.x, b.x, &e);
	 e += (a.y + b.y);
	 s = quick_two_sum(s, e, &e);
	 return make_dd(s, e);
}

gdd_real add_HH(gdd_real a, gdd_real b) {
	 return sloppy_add(a, b);
}

// Subtractions

gdd_real sub_HH(gdd_real a, gdd_real b) {

	 double s, e;
	 s = two_diff(a.x, b.x, &e);
	 e += a.y;
	 e -= b.y;
	 s = quick_two_sum(s, e, &e);
	 return make_dd(s, e);
}

// double-double - double
gdd_real sub_HD(gdd_real a, double b) {
	 double s1, s2;
	 s1 = two_diff(a.x, b, &s2);
	 s2 += a.y;
	 s1 = quick_two_sum(s1, s2, &s2);
	 return make_dd(s1, s2);
}

// double - double-double
gdd_real sub_DH(double a, gdd_real b) {
	 double s1, s2;
	 s1 = two_diff(a, b.x, &s2);
	 s2 -= b.y;
	 s1 = quick_two_sum(s1, s2, &s2);
	 return make_dd(s1, s2);
}

// Squaring
__attribute__((overloadable))
gdd_real sqr(gdd_real a) {
	 double p1, p2;
	 double s1, s2;
	 p1 = two_sqr(a.x, &p2);
	 //p2 += (2.0 * a.x * a.y);
	 p2 = __dadd_rn(p2, __dmul_rn(__dmul_rn(2.0, a.x), a.y));
	 //p2 += (a.y * a.y);
	 p2 = __dadd_rn(p2, __dmul_rn(a.y, a.y));
	 s1 = quick_two_sum(p1, p2, &s2);
	 return make_dd(s1, s2);
}

__attribute__((overloadable))
gdd_real sqr(double a) {
	 double p1, p2;
	 p1 = two_sqr(a, &p2);
	 return make_dd(p1, p2);
}

/****************** Multiplication ********************/

// double-double * (2.0 ^ exp)
__attribute__((overloadable)) 
gdd_real ldexp(gdd_real a, int exp) {
	 return make_dd(ldexp(a.x, exp), ldexp(a.y, exp));
}

// double-double * double,  where double is a power of 2.
gdd_real mul_pwr2(gdd_real a, double b) {
	 return make_dd(a.x * b, a.y * b);
}

// double-double * double-double
gdd_real mul_HH(gdd_real a, gdd_real b) {
	 double p1, p2;

	 p1 = two_prod(a.x, b.x, &p2);
	 p2 += (a.x * b.y + a.y * b.x);
	 //p2 += __dadd_rn(__dmul_rn(a.x, b.y), __dmul_rn(a.y, b.x));

	 p1 = quick_two_sum(p1, p2, &p2);

	 return make_dd(p1, p2);
}

// double-double * double
gdd_real mul_HD(gdd_real a, double b) {
	 double p1, p2;

	 p1 = two_prod(a.x, b, &p2);
	 p2 = __dadd_rn(p2, (__dmul_rn(a.y, b)));
	 p1 = quick_two_sum(p1, p2, &p2);
	 return make_dd(p1, p2);
}

// double * double-double
gdd_real mul_DH(double a, gdd_real b) {
	 return mul_HD(b, a);
	 //return mul_HH(make_dd(a), b);//can also use
}

/******************* Division *********************/

gdd_real sloppy_div(gdd_real a, gdd_real b) {
	 double s1, s2;
	 double q1, q2;
	 gdd_real r;

	 q1 = a.x / b.x; // approximate quotient

	 // compute  this - q1 * dd
	 r = mul_HD(b, q1);
	 s1 = two_diff(a.x, r.x, &s2);
	 s2 -= r.y;
	 s2 += a.y;

	 /* get next approximation */
	 q2 = (s1 + s2) / b.x;

	 /* renormalize */
	 r.x = quick_two_sum(q1, q2, &s1);
	 r.y = s1;
	 return r;
}

// double-double / double-double
gdd_real div_HH(gdd_real a, gdd_real b) {
	 return sloppy_div(a, b);
}

// double-double / double
gdd_real div_HD(gdd_real a, double b) {

	 double q1, q2;
	 double p1, p2;
	 double s, e;
	 gdd_real r;

	 q1 = a.x / b; /* approximate quotient. */

	 /* Compute  this - q1 * d */
	 p1 = two_prod(q1, b, &p2);
	 s = two_diff(a.x, p1, &e);
	 e = e + a.y;
	 e = e - p2;

	 /* get next approximation. */
	 q2 = (s + e) / b;

	 /* renormalize */
	 r.x = quick_two_sum(q1, q2, &e);
	 r.y = e;

	 return r;
}

// double / double-double
gdd_real div_DH(double a, gdd_real b) {
	 return div_HH(make_dd(a), b);
}

gdd_real inv(gdd_real a) {
	 return div_DH(1.0, a);
}

bool is_zero(gdd_real a) {
	 return (a.x == 0.0);
}

bool is_one(gdd_real a) {
	 return (a.x == 1.0 && a.y == 0.0);
}

/*  this > 0 */
bool is_positive(gdd_real a) {
	 return (a.x > 0.0);
}

/* this < 0 */
bool is_negative(gdd_real a) {
	 return (a.x < 0.0);
}

/* Cast to double. */
double to_double(gdd_real a) {
	 return a.x;
}

float to_float(const gdd_real a)
{
	return a.x;
}

// Comparison

// double-double <= double-double
/*
bool operator<=(const gdd_real &a, const gdd_real &b) {
	 return (a.x < b.x || (a.x == b.x && a.y <= b.y));
}

// Equality Comparisons

// double-double == double
bool operator==(const gdd_real &a, double b) {
	 return (a.x == b && a.y == 0.0);
}

// double-double == double-double
bool operator==(const gdd_real &a, const gdd_real &b) {
	 return (a.x == b.x && a.y == b.y);
}

// double == double-double
bool operator==(double a, const gdd_real &b) {
	 return (a == b.x && b.y == 0.0);
}

// Greater-Than Comparisons

// double-double > double
bool operator>(const gdd_real &a, double b) {
	 return (a.x > b || (a.x == b && a.y > 0.0));
}

// double-double > double-double
bool operator>(const gdd_real &a, const gdd_real &b) {
	 return (a.x > b.x || (a.x == b.x && a.y > b.y));
}

// double > double-double
bool operator>(double a, const gdd_real &b) {
	 return (a > b.x || (a == b.x && b.y < 0.0));
}

// Less-Than Comparisons

// double-double < double
bool operator<(const gdd_real &a, double b) {
	 return (a.x < b || (a.x == b && a.y < 0.0));
}

// double-double < double-double
bool operator<(const gdd_real &a, const gdd_real &b) {
	 return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

// double < double-double
bool operator<(double a, const gdd_real &b) {
	 return (a < b.x || (a == b.x && b.y > 0.0));
}

// Greater-Than-Or-Equal-To Comparisons

// double-double >= double
bool operator>=(const gdd_real &a, double b) {
	 return (a.x > b || (a.x == b && a.y >= 0.0));
}

// double-double >= double-double
bool operator>=(const gdd_real &a, const gdd_real &b) {
	 return (a.x > b.x || (a.x == b.x && a.y >= b.y));
}

// double >= double-double
bool operator>=(double a, const gdd_real &b) {
//  return (b <= a);
//}

// Less-Than-Or-Equal-To Comparisons

// double-double <= double
bool operator<=(const gdd_real &a, double b) {
	 return (a.x < b || (a.x == b && a.y <= 0.0));
}

// double >= double-double
bool operator>=(double a, const gdd_real &b) {
	 return (b <= a);
}

// double-double <= double-double
bool operator<=(const gdd_real &a, const gdd_real &b) {
//  return (a.x[0] < b.x[0] || (a.x[0] == b.x[0] && a.x[1] <= b.x[1]));
//}

// double <= double-double
bool operator<=(double a, const gdd_real &b) {
	 return (b >= a);
}

// Not-Equal-To Comparisons

// double-double != double
bool operator!=(const gdd_real &a, double b) {
	 return (a.x != b || a.y != 0.0);
}

// double-double != double-double
bool operator!=(const gdd_real &a, const gdd_real &b) {
	 return (a.x != b.x || a.y != b.y);
}

// double != double-double
bool operator!=(double a, const gdd_real &b) {
	 return (a != b.x || b.y != 0.0);
}*/

bool gt(gdd_real a, gdd_real b)//operator>
{
	 return (a.x > b.x || (a.x == b.x && a.y > b.y));
}

gdd_real nint(gdd_real a) {
	 double hi = nint(a.x);
	 double lo;

	 if (hi == a.x) {
		  /* High word is an integer already.  Round the low word.*/
		  lo = nint(a.y);

		  /* Renormalize. This is needed if x[0] = some integer, x[1] = 1/2.*/
		  hi = quick_two_sum(hi, lo, &lo);
	 } else {
		  /* High word is not an integer. */
		  lo = 0.0;
		  if (fabs(hi - a.x) == 0.5 && a.y < 0.0) {
				/* There is a tie in the high word, consult the low word 
					to break the tie. */
				hi -= 1.0; /* NOTE: This does not cause INEXACT. */
		  }
	 }

	 return make_dd(hi, lo);
}

__attribute__((overloadable))
gdd_real abs(gdd_real a) {
	 return (a.x < 0.0) ? negative(a) : a;
}

__attribute__((overloadable))
gdd_real fabs(gdd_real a) {
	 return abs(a);
}

//extra operators
/*
gdd_real operator+=(gdd_real &a, const gdd_real &b) {
	 return a = a + b;//sloppy_add(a, b);
}

gdd_real operator-=(gdd_real &a, const gdd_real &b) {
	 return a = a - b;
}

gdd_real operator*=(gdd_real &a, const gdd_real &b) {
	 return a = a * b;
}

gdd_real operator*=(gdd_real &a, const double &b) {
	 return a = a * b;
}*/
";
		public static string strDD128Trig =
@"
#define CosAsSin
//#define UseTan
// constants
#define _dd_eps (4.93038065763132e-32)  // 2^-104
#define _dd_e make_dd(2.718281828459045091e+00, 1.445646891729250158e-16)
#define _dd_log2 make_dd(6.931471805599452862e-01, 2.319046813846299558e-17)
#define _dd_2pi make_dd(6.283185307179586232e+00, 2.449293598294706414e-16)
#define _dd_pi make_dd(3.141592653589793116e+00, 1.224646799147353207e-16)
#define _dd_pi2 make_dd(1.570796326794896558e+00, 6.123233995736766036e-17)
#define _dd_pi16 make_dd(1.963495408493620697e-01, 7.654042494670957545e-18)
#define _dd_pi4 make_dd(7.853981633974482790e-01, 3.061616997868383018e-17)
#define _dd_3pi4 make_dd(2.356194490192344837e+00, 9.1848509936051484375e-17)
#define Pi4    _dd_pi4

#define n_dd_inv_fact (15)
static __constant gdd_real dd_inv_fact[n_dd_inv_fact] =
{
	{1.66666666666666657e-01, 9.25185853854297066e-18},
	{4.16666666666666644e-02, 2.31296463463574266e-18},
	{8.33333333333333322e-03, 1.15648231731787138e-19},
	{1.38888888888888894e-03, -5.30054395437357706e-20},
	{1.98412698412698413e-04, 1.72095582934207053e-22},
	{2.48015873015873016e-05, 2.15119478667758816e-23},
	{2.75573192239858925e-06, -1.85839327404647208e-22},
	{2.75573192239858883e-07, 2.37677146222502973e-23},
	{2.50521083854417202e-08, -1.44881407093591197e-24},
	{2.08767569878681002e-09, -1.20734505911325997e-25},
	{1.60590438368216133e-10, 1.25852945887520981e-26},
	{1.14707455977297245e-11, 2.06555127528307454e-28},
	{7.64716373181981641e-13, 7.03872877733453001e-30},
	{4.77947733238738525e-14, 4.39920548583408126e-31},
	{2.81145725434552060e-15, 1.65088427308614326e-31}
};

static __constant gdd_real d_sin_table_dd[4] =
{
	{1.950903220161282758e-01, -7.991079068461731263e-18},
	{3.826834323650897818e-01, -1.005077269646158761e-17},
	{5.555702330196021776e-01, 4.709410940561676821e-17},
	{7.071067811865475727e-01, -4.833646656726456726e-17}
};

static __constant gdd_real d_cos_table_dd[4] =
{
	{9.807852804032304306e-01, 1.854693999782500573e-17},
	{9.238795325112867385e-01, 1.764504708433667706e-17},
	{8.314696123025452357e-01, 1.407385698472802389e-18},
	{7.071067811865475727e-01, -4.833646656726456726e-17}
};

/* Computes the square root of the double-double number dd.
	NOTE: dd must be a non-negative number.                   */
__attribute__((overloadable))
gdd_real sqrt(gdd_real a) {
	 if (is_zero(a))
		  return make_dd(0.0);

	 //TODO: should make an error
	 if (is_negative(a)) {
		  //return _nan;
		  return make_dd(0.0);
	 }

	 double x = 1.0 / sqrt(a.x);
	 double ax = a.x * x;

	 return dd_add(ax, HP(a - sqr(ax)).x * (x * 0.5));
}

gdd_real sin_taylor(gdd_real a)
{
	 const double thresh = 0.5 * fabs(to_double(a)) * _dd_eps;
	 gdd_real r, s, t, x;

	//not use(we are NOT on CPU) if (is_zero(a)) return make_dd(0.0);

	 int i = 0;
	 x = negative(sqr(a)); //-sqr(a);
	 s = a;
	 r = a;
	 do {
		  r = HP(r * x);
		  t = HP(r * dd_inv_fact[i]);
		  s = HP(s + t);
		  i += 2;
	 } while (i < n_dd_inv_fact && fabs(to_double(t)) > thresh);

	 return s;
}

gdd_real cos_taylor(gdd_real a)
{
	 const double thresh = 0.5 * _dd_eps;
	 gdd_real r, s, t, x;
	 int i = 1;

	//not use(we are NOT on CPU) if (is_zero(a)) return make_dd(1.0);

	 x = negative(sqr(a));
	 r = x;
	 s = add_DH(1.0, mul_pwr2(r, 0.5));
	 do {
		  r = HP(r * x);
		  t = HP(r * dd_inv_fact[i]);
		  s = HP(s + t);
		  i += 2;
	 } while (i < n_dd_inv_fact && fabs(to_double(t)) > thresh);

	 return s;
}

void sincos_taylor(gdd_real a, gdd_real *sin_a, gdd_real *cos_a)
{
	//not use(we are NOT on CPU)
	/*if (is_zero(a))
	{
		*sin_a = make_dd(0.0);
		*cos_a = make_dd(1.0);
		return;
	}*/

	 *sin_a = sin_taylor(a);
	 *cos_a = sqrt(sub_DH(1.0, sqr(*sin_a)));
}

__attribute__((overloadable))
gdd_real sin(gdd_real a)
{
	//not use(we are NOT on CPU)if (is_zero(a)) return make_dd(0.0);

	 // approximately reduce modulo 2*pi
	 gdd_real z = nint(HP(a / _dd_2pi));
	 gdd_real r = HP(a - _dd_2pi * z);

	 // approximately reduce modulo pi/2 and then modulo pi/16.
	 gdd_real t;
	 double q = floor(r.x / _dd_pi2.x + 0.5);
	 t = HP(r - mul_HD(_dd_pi2, q));
	 int j = (int) (q);
	 q = floor(t.x / _dd_pi16.x + 0.5);
	 t = HP(t - mul_HD(_dd_pi16, q));
	 int k = (int) (q);
	 int abs_k = abs(k);

	 if (j < -2 || j > 2) {
		  //dd_real::error(""(dd_real::sin): Cannot reduce modulo pi/2."");
		  r.x = r.y = 0.0;
		  return r;
	 }

	 if (abs_k > 4) {
		  //dd_real::error(""(dd_real::sin): Cannot reduce modulo pi/16."");
		  r.x = r.y = 0.0;
		  return r;
	 }

	 if (k == 0) {
		  switch (j) {
				case 0:
					 return sin_taylor(t);
				case 1:
					 return cos_taylor(t);
				case -1:
					 return negative(cos_taylor(t));
				default:
					 return negative(sin_taylor(t));
		  }
	 }

	 gdd_real u = d_cos_table_dd[abs_k - 1];
	 gdd_real v = d_sin_table_dd[abs_k - 1];
	 gdd_real sin_t, cos_t;
	 sincos_taylor(t, &sin_t, &cos_t);
	 if (j == 0) {
		  if (k > 0) {
				r = HP(u * sin_t + v * cos_t);
		  } else {
				r = HP(u * sin_t - v * cos_t);
		  }
	 } else if (j == 1) {
		  if (k > 0) {
				r = HP(u * cos_t - v * sin_t);
		  } else {
				r = HP(u * cos_t + v * sin_t);
		  }
	 } else if (j == -1) {
		  if (k > 0) {
				r = HP(v * sin_t - u * cos_t);
		  } else if (k < 0) {
				r = HP(negative(u) * cos_t - v * sin_t);
		  }
	 } else {
		  if (k > 0) {
				r = HP(negative(u) * sin_t - v * cos_t);
		  } else {
				r = HP(v * cos_t - u * sin_t);
		  }
	 }

	 return r;
}

__attribute__((overloadable))
gdd_real cos(gdd_real a)
{
#ifdef CosAsSin
	return sin(add_HH(a, _dd_pi2));
#else
	//not use(we are NOT on CPU) if (is_zero(a)) return make_dd(1.0);

	 // approximately reduce modulo 2*pi
	 gdd_real z = nint(HP(a / _dd_2pi));
	 gdd_real r = HP(a - z * _dd_2pi);

	 // approximately reduce modulo pi/2 and then modulo pi/16
	 gdd_real t;
	 double q = floor(r.x / _dd_pi2.x + 0.5);
	 t = HP(r - mul_HD(_dd_pi2, q));
	 int j = (int) (q);
	 q = floor(t.x / _dd_pi16.x + 0.5);
	 t = HP(t - mul_HD(_dd_pi16, q));
	 int k = (int) (q);
	 int abs_k = abs(k);

	 if (j < -2 || j > 2) {
		  //dd_real::error(""(dd_real::cos): Cannot reduce modulo pi/2."");
		  //return dd_real::_nan;
		  return make_dd(0.0);
	 }

	 if (abs_k > 4) {
		  //dd_real::error(""(dd_real::cos): Cannot reduce modulo pi/16."");
		  //return dd_real::_nan;
		  return make_dd(0.0);
	 }

	 if (k == 0) {
		  switch (j) {
				case 0:
					 return cos_taylor(t);
				case 1:
					 return negative(sin_taylor(t));
				case -1:
					 return sin_taylor(t);
				default:
					 return negative(cos_taylor(t));
		  }
	 }

	 gdd_real sin_t, cos_t;
	 sincos_taylor(t, &sin_t, &cos_t);
	 gdd_real u = d_cos_table_dd[abs_k - 1];
	 gdd_real v = d_sin_table_dd[abs_k - 1];

	 if (j == 0) {
		  if (k > 0) {
				r = HP(u * cos_t - v * sin_t);
		  } else {
				r = HP(u * cos_t + v * sin_t);
		  }
	 } else if (j == 1) {
		  if (k > 0) {
				r = HP(negative(u) * sin_t - v * cos_t);
		  } else {
				r = HP(v * cos_t - u * sin_t);
		  }
	 } else if (j == -1) {
		  if (k > 0) {
				r = HP(u * sin_t + v * cos_t);
		  } else {
				r = HP(u * sin_t - v * cos_t);
		  }
	 } else {
		  if (k > 0) {
				r = HP(v * sin_t - u * cos_t);
		  } else {
				r = HP(negative(u) * cos_t - v * sin_t);
		  }
	 }

	 return r;
#endif
}
#ifdef UseTan
void sincos(gdd_real a, gdd_real *sin_a, gdd_real *cos_a)
{
	//not use(we are NOT on CPU)
	/*if (is_zero(a))
	{
		*sin_a = make_dd(0.0);
		*cos_a = make_dd(1.0);
		return;
	}*/

	 // approximately reduce modulo 2*pi
	 gdd_real z = nint(HP(a / _dd_2pi));
	 gdd_real r = HP(a - _dd_2pi * z);

	 // approximately reduce module pi/2 and pi/16
	 gdd_real t;
	 double q = floor(r.x / _dd_pi2.x + 0.5);
	 t = HP(r - mul_HD(_dd_pi2, q));
	 int j = (int) (q);
	 int abs_j = abs(j);
	 q = floor(t.x / _dd_pi16.x + 0.5);
	 t = HP(t - mul_HD(_dd_pi16, q));
	 int k = (int) (q);
	 int abs_k = abs(k);

	 if (abs_j > 2) {
		  //dd_real::error(""(dd_real::sincos): Cannot reduce modulo pi/2."");
		  //cos_a = sin_a = dd_real::_nan;
		  *cos_a = *sin_a = make_dd(0.0);
		  return;
	 }

	 if (abs_k > 4) {
		  //dd_real::error(""(dd_real::sincos): Cannot reduce modulo pi/16."");
		  //cos_a = sin_a = dd_real::_nan;
		  *cos_a = *sin_a = make_dd(0.0);
		  return;
	 }

	 gdd_real sin_t, cos_t;
	 gdd_real s, c;

	 sincos_taylor(t, &sin_t, &cos_t);

	 if (abs_k == 0) {
		  s = sin_t;
		  c = cos_t;
	 } else {
		  gdd_real u = d_cos_table_dd[abs_k - 1];
		  gdd_real v = d_sin_table_dd[abs_k - 1];

		  if (k > 0) {
				s = HP(u * sin_t + v * cos_t);
				c = HP(u * cos_t - v * sin_t);
		  } else {
				s = HP(u * sin_t - v * cos_t);
				c = HP(u * cos_t + v * sin_t);
		  }
	 }

	 if (abs_j == 0) {
		  *sin_a = s;
		  *cos_a = c;
	 } else if (j == 1) {
		  *sin_a = c;
		  *cos_a = negative(s);
	 } else if (j == -1) {
		  *sin_a = negative(c);
		  *cos_a = s;
	 } else {
		  *sin_a = negative(s);
		  *cos_a = negative(c);
	 }
}

__attribute__((overloadable))
gdd_real tan(gdd_real a)
{
	 gdd_real s, c;
	 sincos(a, &s, &c);
	 return HP(s / c);
}
#endif
";
		public static string strTypeDefQD256 =
@"
//#define UseDouble4
#ifdef UseDouble4
typedef double4 gqd_real;
#else
typedef struct __attribute__ ((packed)) dbl4
{
	double x,y,z,w;
} gqd_real;
#endif
typedef gqd_real QD256;
";//from gdd_basic.cu
		public static string strQD256 = strTypeDefQD256 +
@"
// type construction
__attribute__((overloadable))
__attribute__((always_inline))
gqd_real make_qd(const double x, const double y, const double z, const double w)
{
#ifdef UseDouble4
	return (gqd_real)(x, y, z, w);//make_double4
#else
	return (gqd_real){x, y, z, w};
#endif
}

__attribute__((overloadable))
__attribute__((always_inline))
gqd_real make_qd(const double x)
{
	#ifdef UseDouble4
	return (gqd_real)(x, 0.0, 0.0, 0.0);
	#else
	return (gqd_real){x, 0.0, 0.0, 0.0};
	#endif
}

#define Zero	make_qd(0.0)
#define One		make_qd(1.0)

// normalization functions
/*void quick_renorm(double *c0, double *c1, double *c2, double *c3, double *c4) {
	 double t0, t1, t2, t3;
	 double s;
	 s = quick_two_sum(*c3, *c4, &t3);
	 s = quick_two_sum(*c2, s, &t2);
	 s = quick_two_sum(*c1, s, &t1);
	 *c0 = quick_two_sum(*c0, s, &t0);

	 s = quick_two_sum(t2, t3, &t2);
	 s = quick_two_sum(t1, s, &t1);
	 *c1 = quick_two_sum(t0, s, &t0);

	 s = quick_two_sum(t1, t2, &t1);
	 *c2 = quick_two_sum(t0, s, &t0);

	 *c3 = t0 + t1;
}*/

__attribute__((overloadable))
void renorm(double *c0, double *c1, double *c2, double *c3) {
	 double s0, s1, s2 = 0.0, s3 = 0.0;

	 s0 = quick_two_sum(*c2, *c3, c3);
	 s0 = quick_two_sum(*c1, s0, c2);
	 *c0 = quick_two_sum(*c0, s0, c1);

	 s0 = *c0;
	 s1 = *c1;
	 if (s1 != 0.0) {
		  s1 = quick_two_sum(s1, *c2, &s2);
		  if (s2 != 0.0)
				s2 = quick_two_sum(s2, *c3, &s3);
		  else
				s1 = quick_two_sum(s1, *c3, &s2);
	 } else {
		  s0 = quick_two_sum(s0, *c2, &s1);
		  if (s1 != 0.0)
				s1 = quick_two_sum(s1, *c3, &s2);
		  else
				s0 = quick_two_sum(s0, *c3, &s1);
	 }

	 *c0 = s0;
	 *c1 = s1;
	 *c2 = s2;
	 *c3 = s3;
}

__attribute__((overloadable))
void renorm(double *c0, double *c1, double *c2, double *c3, double c4) {
	 double s0, s1, s2 = 0.0, s3 = 0.0;

	 s0 = quick_two_sum(*c3, c4, &c4);
	 s0 = quick_two_sum(*c2, s0, c3);
	 s0 = quick_two_sum(*c1, s0, c2);
	 *c0 = quick_two_sum(*c0, s0, c1);

	 s0 = *c0;
	 s1 = *c1;

	 s0 = quick_two_sum(*c0, *c1, &s1);
	 if (s1 != 0.0) {
		  s1 = quick_two_sum(s1, *c2, &s2);
		  if (s2 != 0.0) {
				s2 = quick_two_sum(s2, *c3, &s3);
				if (s3 != 0.0)
					 s3 += c4;
				else
					 s2 += c4;
		  } else {
				s1 = quick_two_sum(s1, *c3, &s2);
				if (s2 != 0.0)
					 s2 = quick_two_sum(s2, c4, &s3);
				else
					 s1 = quick_two_sum(s1, c4, &s2);
		  }
	 } else {
		  s0 = quick_two_sum(s0, *c2, &s1);
		  if (s1 != 0.0) {
				s1 = quick_two_sum(s1, *c3, &s2);
				if (s2 != 0.0)
					 s2 = quick_two_sum(s2, c4, &s3);
				else
					 s1 = quick_two_sum(s1, c4, &s2);
		  } else {
				s0 = quick_two_sum(s0, *c3, &s1);
				if (s1 != 0.0)
					 s1 = quick_two_sum(s1, c4, &s2);
				else
					 s0 = quick_two_sum(s0, c4, &s1);
		  }
	 }

	 *c0 = s0;
	 *c1 = s1;
	 *c2 = s2;
	 *c3 = s3;
}

/*__attribute__((overloadable))
void renorm(gqd_real *x) {
	 renorm(*x.x, *x.y, *x.z, *x.w);
}

__attribute__((overloadable))
void renorm(gqd_real *x, double e) {
	 renorm(*x.x, *x.y, *x.z, *x.w, e);
}*/

/** additions */
void three_sum(double *a, double *b, double *c) {
	 double t1, t2, t3;
	 t1 = two_sum(*a, *b, &t2);
	 *a = two_sum(*c, t1, &t3);
	 *b = two_sum(t2, t3, c);
}

void three_sum2(double *a, double *b, double c) {
	 double t1, t2, t3;
	 t1 = two_sum(*a, *b, &t2);
	 *a = two_sum(c, t1, &t3);
	 *b = (t2 + t3);
}

///qd = qd + double
gqd_real add_HD(const gqd_real a, double b)//operator+
{
	 double c0, c1, c2, c3;
	 double e;

	 c0 = two_sum(a.x, b, &e);
	 c1 = two_sum(a.y, e, &e);
	 c2 = two_sum(a.z, e, &e);
	 c3 = two_sum(a.w, e, &e);

	 renorm(&c0, &c1, &c2, &c3, e);

	 return make_qd(c0, c1, c2, c3);
}

///qd = double + qd
gqd_real add_DH(double a, const gqd_real b)//operator+
{
	 return add_HD(b, a);
}

///qd = qd + qd

gqd_real sloppy_add(const gqd_real a, const gqd_real b) {
	 double s0, s1, s2, s3;
	 double t0, t1, t2, t3;

	 double v0, v1, v2, v3;
	 double u0, u1, u2, u3;
	 double w0, w1, w2, w3;

	 s0 = a.x + b.x;
	 s1 = a.y + b.y;
	 s2 = a.z + b.z;
	 s3 = a.w + b.w;

	 v0 = s0 - a.x;
	 v1 = s1 - a.y;
	 v2 = s2 - a.z;
	 v3 = s3 - a.w;

	 u0 = s0 - v0;
	 u1 = s1 - v1;
	 u2 = s2 - v2;
	 u3 = s3 - v3;

	 w0 = a.x - u0;
	 w1 = a.y - u1;
	 w2 = a.z - u2;
	 w3 = a.w - u3;

	 u0 = b.x - v0;
	 u1 = b.y - v1;
	 u2 = b.z - v2;
	 u3 = b.w - v3;

	 t0 = w0 + u0;
	 t1 = w1 + u1;
	 t2 = w2 + u2;
	 t3 = w3 + u3;

	 s1 = two_sum(s1, t0, &t0);
	 three_sum(&s2, &t0, &t1);
	 three_sum2(&s3, &t0, t2);
	 t0 = t0 + t1 + t3;

	 renorm(&s0, &s1, &s2, &s3, t0);

	 return make_qd(s0, s1, s2, s3);
}

gqd_real add_HH(const gqd_real a, const gqd_real b)//operator+
{
	 return sloppy_add(a, b);
}

/** subtractions */
gqd_real negative(const gqd_real a) {
	 return make_qd(-a.x, -a.y, -a.z, -a.w);
}

/*gqd_real operator-(const gqd_real &a)
{
	return negative(a);
}*/

gqd_real sub_HD(const gqd_real a, double b)//operator-
{
	 return add_HD(a, -b);
}

gqd_real sub_DH(double a, const gqd_real b)//operator-
{
	 return add_DH(a, negative(b));
}

gqd_real sub_HH(const gqd_real a, const gqd_real b)//operator-
{
	 return add_HH(a, negative(b));
}

/** multiplications */
gqd_real mul_pwr2(const gqd_real a, double b) {
	 return make_qd(a.x * b, a.y * b, a.z * b, a.w * b);
}


//quad_double * double

gqd_real mul_HD(const gqd_real a, double b)//operator*
{
	 double p0, p1, p2, p3;
	 double q0, q1, q2;
	 double s0, s1, s2, s3, s4;

	 p0 = two_prod(a.x, b, &q0);
	 p1 = two_prod(a.y, b, &q1);
	 p2 = two_prod(a.z, b, &q2);
	 p3 = a.w * b;

	 s0 = p0;

	 s1 = two_sum(q0, p1, &s2);

	 three_sum(&s2, &q1, &p2);

	 three_sum2(&q1, &q2, p3);
	 s3 = q1;

	 s4 = q2 + p2;

	 renorm(&s0, &s1, &s2, &s3, s4);
	 return make_qd(s0, s1, s2, s3);
}
//quad_double = double*quad_double

gqd_real mul_DH(double a, const gqd_real b)//operator*
{
	 return mul_HD(b, a);
}

gqd_real sloppy_mul(const gqd_real a, const gqd_real b) {
	 double p0, p1, p2, p3, p4, p5;
	 double q0, q1, q2, q3, q4, q5;
	 double t0, t1;
	 double s0, s1, s2;

	 p0 = two_prod(a.x, b.x, &q0);

	 p1 = two_prod(a.x, b.y, &q1);
	 p2 = two_prod(a.y, b.x, &q2);

	 p3 = two_prod(a.x, b.z, &q3);
	 p4 = two_prod(a.y, b.y, &q4);
	 p5 = two_prod(a.z, b.x, &q5);


	 /* Start Accumulation */
	 three_sum(&p1, &p2, &q0);

	 /* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	 three_sum(&p2, &q1, &q2);
	 three_sum(&p3, &p4, &p5);
	 /* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
	 s0 = two_sum(p2, p3, &t0);
	 s1 = two_sum(q1, p4, &t1);
	 s2 = q2 + p5;
	 s1 = two_sum(s1, t0, &t0);
	 s2 += (t0 + t1);

	 /* O(eps^3) order terms */
	 //!!!s1 = s1 + (a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x + q0 + q3 + q4 + q5);

	 s1 = s1 + (__dmul_rn(a.x, b.w) + __dmul_rn(a.y, b.z) +
				__dmul_rn(a.z, b.y) + __dmul_rn(a.w, b.x) + q0 + q3 + q4 + q5);
	 renorm(&p0, &p1, &s0, &s1, s2);

	 return make_qd(p0, p1, s0, s1);
}

gqd_real mul_HH(const gqd_real a, const gqd_real b)//operator*
{
	 return sloppy_mul(a, b);
}

gqd_real sqr(const gqd_real a) {
	 double p0, p1, p2, p3, p4, p5;
	 double q0, q1, q2, q3;
	 double s0, s1;
	 double t0, t1;

	 p0 = two_sqr(a.x, &q0);
	 p1 = two_prod(2.0 * a.x, a.y, &q1);
	 p2 = two_prod(2.0 * a.x, a.z, &q2);
	 p3 = two_sqr(a.y, &q3);

	 p1 = two_sum(q0, p1, &q0);

	 q0 = two_sum(q0, q1, &q1);
	 p2 = two_sum(p2, p3, &p3);

	 s0 = two_sum(q0, p2, &t0);
	 s1 = two_sum(q1, p3, &t1);

	 s1 = two_sum(s1, t0, &t0);
	 t0 += t1;

	 s1 = quick_two_sum(s1, t0, &t0);
	 p2 = quick_two_sum(s0, s1, &t1);
	 p3 = quick_two_sum(t1, t0, &q0);

	 p4 = 2.0 * a.x * a.w;
	 p5 = 2.0 * a.y * a.z;

	 p4 = two_sum(p4, p5, &p5);
	 q2 = two_sum(q2, q3, &q3);

	 t0 = two_sum(p4, q2, &t1);
	 t1 = t1 + p5 + q3;

	 p3 = two_sum(p3, t0, &p4);
	 p4 = p4 + q0 + t1;

	 renorm(&p0, &p1, &p2, &p3, p4);
	 return make_qd(p0, p1, p2, p3);
}

/** divisions */
gqd_real sloppy_div(const gqd_real a, const gqd_real b) {
	 double q0, q1, q2, q3;

	 gqd_real r;

	 q0 = a.x / b.x;
	 r = sub_HH(a, mul_HD(b, q0));

	 q1 = r.x / b.x;
	 r = sub_HH(r, mul_HD(b, q1));

	 q2 = r.x / b.x;
	 r = sub_HH(r, mul_HD(b, q2));

	 q3 = r.x / b.x;

	 renorm(&q0, &q1, &q2, &q3);

	 return make_qd(q0, q1, q2, q3);
}

gqd_real div_HH(const gqd_real a, const gqd_real b)//operator/
{
	 return sloppy_div(a, b);
}

/* double / quad-double */
gqd_real div_DH(double a, const gqd_real b)//operator/
{
	 return div_HH(make_qd(a), b);
}

/* quad-double / double */
gqd_real div_HD(const gqd_real a, double b)//operator/
{
	 return div_HH(a, make_qd(b));
}

/********** Miscellaneous **********/
__attribute__((overloadable))
gqd_real abs(const gqd_real a) {
	 return (a.x < 0.0) ? (negative(a)) : (a);
}

/********************** Simple Conversion ********************/
double to_double(const gqd_real a) {
	 return a.x;
}

float to_float(const gqd_real a)
{
	return a.x;
}

__attribute__((overloadable))
gqd_real ldexp(const gqd_real a, int n) {
	 return make_qd(ldexp(a.x, n), ldexp(a.y, n),
				ldexp(a.z, n), ldexp(a.w, n));
}

gqd_real inv(const gqd_real qd) {
	 return div_DH(1.0, qd);
}

//********* Greater-Than Comparison ***********
/*
bool operator>=(const gqd_real a, const gqd_real b) {
	 return (a.x > b.x ||
				(a.x == b.x && (a.y > b.y ||
				(a.y == b.y && (a.z > b.z ||
				(a.z == b.z && a.w >= b.w))))));
}
*/
/********** Greater-Than-Or-Equal-To Comparison **********/
/*
bool operator>=(const gqd_real &a, double b) {
  return (a.x > b || (a.x == b && a.y >= 0.0));
}

bool operator>=(double a, const gqd_real &b) {
  return (b <= a);
}

bool operator>=(const gqd_real &a, const gqd_real &b) {
  return (a.x > b.x || 
			 (a.x == b.x && (a.y > b.y ||
									 (a.y == b.y && (a.z > b.z ||
															 (a.z == b.z && a.w >= b.w))))));
}
 */

/********** Less-Than Comparison ***********/
/*
bool operator<(const gqd_real &a, double b) {
	 return (a.x < b || (a.x == b && a.y < 0.0));
}

bool operator<(const gqd_real &a, const gqd_real &b) {
	 return (a.x < b.x ||
				(a.x == b.x && (a.y < b.y ||
				(a.y == b.y && (a.z < b.z ||
				(a.z == b.z && a.w < b.w))))));
}

bool operator<=(const gqd_real &a, const gqd_real &b) {
	 return (a.x < b.x ||
				(a.x == b.x && (a.y < b.y ||
				(a.y == b.y && (a.z < b.z ||
				(a.z == b.z && a.w <= b.w))))));
}

bool operator==(const gqd_real &a, const gqd_real &b) {
	 return (a.x == b.x && a.y == b.y &&
				a.z == b.z && a.w == b.w);
}
*/
/********** Less-Than-Or-Equal-To Comparison **********/
/*
bool operator<=(const gqd_real &a, double b) {
	 return (a.x < b || (a.x == b && a.y <= 0.0));
}
*/


/********** Greater-Than-Or-Equal-To Comparison **********/
/*
bool operator>=(const gqd_real &a, double b) {
	 return (a.x > b || (a.x == b && a.y >= 0.0));
}

bool operator<=(double a, const gqd_real &b) {
	 return (b >= a);
}

bool operator>=(double a, const gqd_real &b) {
	 return (b <= a);
}
*/


/********** Greater-Than Comparison ***********/
/*
bool operator>(const gqd_real &a, double b) {
	 return (a.x > b || (a.x == b && a.y > 0.0));
}

bool operator<(double a, const gqd_real &b) {
	 return (b > a);
}

bool operator>(double a, const gqd_real &b) {
	 return (b < a);
}
*/
bool gt(const gqd_real a, const gqd_real b)//operator>
{
	 return (a.x > b.x ||
				(a.x == b.x && (a.y > b.y ||
				(a.y == b.y && (a.z > b.z ||
				(a.z == b.z && a.w > b.w))))));
}

bool is_zero(const gqd_real x) {
	 return (x.x == 0.0);
}

bool is_one(const gqd_real x) {
	 return (x.x == 1.0 && x.y == 0.0 && x.z == 0.0 && x.w == 0.0);
}

bool is_positive(const gqd_real x) {
	 return (x.x > 0.0);
}

bool is_negative(const gqd_real x) {
	 return (x.x < 0.0);
}

gqd_real nint(const gqd_real a) {
	 double x0, x1, x2, x3;

	 x0 = nint(a.x);
	 x1 = x2 = x3 = 0.0;

	 if (x0 == a.x) {
		  /* First double is already an integer. */
		  x1 = nint(a.y);

		  if (x1 == a.y) {
				/* Second double is already an integer. */
				x2 = nint(a.z);

				if (x2 == a.z) {
					 /* Third double is already an integer. */
					 x3 = nint(a.w);
				} else {
					 if (fabs(x2 - a.z) == 0.5 && a.w < 0.0) {
						  x2 -= 1.0;
					 }
				}

		  } else {
				if (fabs(x1 - a.y) == 0.5 && a.z < 0.0) {
					 x1 -= 1.0;
				}
		  }

	 } else {
		  /* First double is not an integer. */
		  if (fabs(x0 - a.x) == 0.5 && a.y < 0.0) {
				x0 -= 1.0;
		  }
	 }

	 renorm(&x0, &x1, &x2, &x3);
	 return make_qd(x0, x1, x2, x3);
}

__attribute__((overloadable))
gqd_real fabs(const gqd_real a) {
	 return abs(a);
}

//extra operators
/*
__inline__
gqd_real operator+=(gqd_real &a, const gqd_real &b) {
	 return a = a + b;//sloppy_add(a, b);
}

__inline__
gqd_real operator-=(gqd_real &a, const gqd_real &b) {
	 return a = a - b;
}

__inline__
gqd_real operator*=(gqd_real &a, const gqd_real &b) {
	 return a = a * b;
}

__inline__
gqd_real operator*=(gqd_real &a, const double &b) {
	 return a = a * b;
}
*/
";
		public static string strQD256Trig =
@"
#define CosAsSin
//#define UseTan
// constants
#define _qd_e make_qd(2.718281828459045091e+00, 1.445646891729250158e-16,  -2.127717108038176765e-33, 1.515630159841218954e-49)
#define _qd_log2 make_qd(6.931471805599452862e-01, 2.319046813846299558e-17,5.707708438416212066e-34,-3.582432210601811423e-50)
#define _qd_eps (1.21543267145725e-63) // = 2^-209
#define _qd_2pi make_qd(6.283185307179586232e+00, 2.449293598294706414e-16, -5.989539619436679332e-33, 2.224908441726730563e-49)
#define _qd_pi make_qd(3.141592653589793116e+00, 1.224646799147353207e-16, -2.994769809718339666e-33, 1.112454220863365282e-49)
#define _qd_pi2 make_qd(1.570796326794896558e+00, 6.123233995736766036e-17, -1.497384904859169833e-33, 5.562271104316826408e-50)
#define _qd_pi1024 make_qd( 3.067961575771282340e-03, 1.195944139792337116e-19,  -2.924579892303066080e-36, 1.086381075061880158e-52)
#define _qd_pi4 make_qd(7.853981633974482790e-01, 3.061616997868383018e-17, -7.486924524295849165e-34, 2.781135552158413204e-50)
#define _qd_3pi4 make_qd(2.356194490192344837e+00, 9.1848509936051484375e-17, 3.9168984647504003225e-33, -2.5867981632704860386e-49)
#define Pi4    _qd_pi4

//static const int n_inv_fact = 15;
#define n_inv_fact (15)
static __constant gqd_real inv_fact[n_inv_fact] =
{
	{1.66666666666666657e-01, 9.25185853854297066e-18, 5.13581318503262866e-34, 2.85094902409834186e-50},
	{4.16666666666666644e-02, 2.31296463463574266e-18, 1.28395329625815716e-34, 7.12737256024585466e-51},
	{8.33333333333333322e-03, 1.15648231731787138e-19, 1.60494162032269652e-36, 2.22730392507682967e-53},
	{1.38888888888888894e-03, -5.30054395437357706e-20, -1.73868675534958776e-36, -1.63335621172300840e-52},
	{1.98412698412698413e-04, 1.72095582934207053e-22, 1.49269123913941271e-40, 1.29470326746002471e-58},
	{2.48015873015873016e-05, 2.15119478667758816e-23, 1.86586404892426588e-41, 1.61837908432503088e-59},
	{2.75573192239858925e-06, -1.85839327404647208e-22, 8.49175460488199287e-39, -5.72661640789429621e-55},
	{2.75573192239858883e-07, 2.37677146222502973e-23, -3.26318890334088294e-40, 1.61435111860404415e-56},
	{2.50521083854417202e-08, -1.44881407093591197e-24, 2.04267351467144546e-41, -8.49632672007163175e-58},
	{2.08767569878681002e-09, -1.20734505911325997e-25, 1.70222792889287100e-42, 1.41609532150396700e-58},
	{1.60590438368216133e-10, 1.25852945887520981e-26, -5.31334602762985031e-43, 3.54021472597605528e-59},
	{1.14707455977297245e-11, 2.06555127528307454e-28, 6.88907923246664603e-45, 5.72920002655109095e-61},
	{7.64716373181981641e-13, 7.03872877733453001e-30, -7.82753927716258345e-48, 1.92138649443790242e-64},
	{4.77947733238738525e-14, 4.39920548583408126e-31, -4.89221204822661465e-49, 1.20086655902368901e-65},
	{2.81145725434552060e-15, 1.65088427308614326e-31, -2.87777179307447918e-50, 4.27110689256293549e-67},
};
static __constant gqd_real d_sin_table_qd[256] =
{
	{3.0679567629659761e-03, 1.2690279085455925e-19, 5.2879464245328389e-36, -1.7820334081955298e-52},
	{6.1358846491544753e-03, 9.0545257482474933e-20, 1.6260113133745320e-37, -9.7492001208767410e-55},
	{9.2037547820598194e-03, -1.2136591693535934e-19,5.5696903949425567e-36, 1.2505635791936951e-52},
	{1.2271538285719925e-02, 6.9197907640283170e-19,-4.0203726713435555e-36, -2.0688703606952816e-52},
	{1.5339206284988102e-02, -8.4462578865401696e-19,4.6535897505058629e-35, -1.3923682978570467e-51},
	{1.8406729905804820e-02, 7.4195533812833160e-19, 3.9068476486787607e-35, 3.6393321292898614e-52},
	{2.1474080275469508e-02, -4.5407960207688566e-19,-2.2031770119723005e-35, 1.2709814654833741e-51},
	{2.4541228522912288e-02, -9.1868490125778782e-20, 4.8706148704467061e-36, -2.8153947855469224e-52},
	{2.7608145778965743e-02, -1.5932358831389269e-18, -7.0475416242776030e-35, -2.7518494176602744e-51},
	{3.0674803176636626e-02, -1.6936054844107918e-20, -2.0039543064442544e-36, -1.6267505108658196e-52},
	{3.3741171851377587e-02, -2.0096074292368340e-18, -1.3548237016537134e-34, 6.5554881875899973e-51},
	{3.6807222941358832e-02, 6.1060088803529842e-19, -4.0448721259852727e-35, -2.1111056765671495e-51},
	{3.9872927587739811e-02, 4.6657453481183289e-19, 3.4119333562288684e-35, 2.4007534726187511e-51},
	{4.2938256934940820e-02, 2.8351940588660907e-18, 1.6991309601186475e-34, 6.8026536098672629e-51},
	{4.6003182130914630e-02, -1.1182813940157788e-18, 7.5235020270378946e-35, 4.1187304955493722e-52},
	{4.9067674327418015e-02, -6.7961037205182801e-19, -4.4318868124718325e-35, -9.9376628132525316e-52},
	{5.2131704680283324e-02, -2.4243695291953779e-18, -1.3675405320092298e-34, -8.3938137621145070e-51},
	{5.5195244349689941e-02, -1.3340299860891103e-18, -3.4359574125665608e-35, 1.1911462755409369e-51},
	{5.8258264500435759e-02, 2.3299905496077492e-19, 1.9376108990628660e-36, -5.1273775710095301e-53},
	{6.1320736302208578e-02, -5.1181134064638108e-19, -4.2726335866706313e-35, 2.6368495557440691e-51},
	{6.4382630929857465e-02, -4.2325997000052705e-18, 3.3260117711855937e-35, 1.4736267706718352e-51},
	{6.7443919563664065e-02, -6.9221796556983636e-18, 1.5909286358911040e-34, -7.8828946891835218e-51},
	{7.0504573389613870e-02, -6.8552791107342883e-18, -1.9961177630841580e-34, 2.0127129580485300e-50},
	{7.3564563599667426e-02, -2.7784941506273593e-18, -9.1240375489852821e-35, -1.9589752023546795e-51},
	{7.6623861392031492e-02, 2.3253700287958801e-19, -1.3186083921213440e-36, -4.9927872608099673e-53},
	{7.9682437971430126e-02, -4.4867664311373041e-18, 2.8540789143650264e-34, 2.8491348583262741e-51},
	{8.2740264549375692e-02, 1.4735983530877760e-18, 3.7284093452233713e-35, 2.9024430036724088e-52},
	{8.5797312344439894e-02, -3.3881893830684029e-18, -1.6135529531508258e-34, 7.7294651620588049e-51},
	{8.8853552582524600e-02, -3.7501775830290691e-18, 3.7543606373911573e-34, 2.2233701854451859e-50},
	{9.1908956497132724e-02, 4.7631594854274564e-18, 1.5722874642939344e-34, -4.8464145447831456e-51},
	{9.4963495329639006e-02, -6.5885886400417564e-18, -2.1371116991641965e-34, 1.3819370559249300e-50},
	{9.8017140329560604e-02, -1.6345823622442560e-18, -1.3209238810006454e-35, -3.5691060049117942e-52},
	{1.0106986275482782e-01, 3.3164325719308656e-18, -1.2004224885132282e-34, 7.2028828495418631e-51},
	{1.0412163387205457e-01, 6.5760254085385100e-18, 1.7066246171219214e-34, -4.9499340996893514e-51},
	{1.0717242495680884e-01, 6.4424044279026198e-18, -8.3956976499698139e-35, -4.0667730213318321e-51},
	{1.1022220729388306e-01, -5.6789503537823233e-19, 1.0380274792383233e-35, 1.5213997918456695e-52},
	{1.1327095217756435e-01, 2.7100481012132900e-18, 1.5323292999491619e-35, 4.9564432810360879e-52},
	{1.1631863091190477e-01, 1.0294914877509705e-18, -9.3975734948993038e-35, 1.3534827323719708e-52},
	{1.1936521481099137e-01, -3.9500089391898506e-18, 3.5317349978227311e-34, 1.8856046807012275e-51},
	{1.2241067519921620e-01, 2.8354501489965335e-18, 1.8151655751493305e-34, -2.8716592177915192e-51},
	{1.2545498341154623e-01, 4.8686751763148235e-18, 5.9878105258097936e-35, -3.3534629098722107e-51},
	{1.2849811079379317e-01, 3.8198603954988802e-18, -1.8627501455947798e-34, -2.4308161133527791e-51},
	{1.3154002870288312e-01, -5.0039708262213813e-18, -1.2983004159245552e-34, -4.6872034915794122e-51},
	{1.3458070850712620e-01, -9.1670359171480699e-18, 1.5916493007073973e-34, 4.0237002484366833e-51},
	{1.3762012158648604e-01, 6.6253255866774482e-18, -2.3746583031401459e-34, -9.3703876173093250e-52},
	{1.4065823933284924e-01, -7.9193932965524741e-18, 6.0972464202108397e-34, 2.4566623241035797e-50},
	{1.4369503315029444e-01, 1.1472723016618666e-17, -5.1884954557576435e-35, -4.2220684832186607e-51},
	{1.4673047445536175e-01, 3.7269471470465677e-18, 3.7352398151250827e-34, -4.0881822289508634e-51},
	{1.4976453467732151e-01, 8.0812114131285151e-18, 1.2979142554917325e-34, 9.9380667487736254e-51},
	{1.5279718525844344e-01, -7.6313573938416838e-18, 5.7714690450284125e-34, -3.7731132582986687e-50},
	{1.5582839765426523e-01, 3.0351307187678221e-18, -1.0976942315176184e-34, 7.8734647685257867e-51},
	{1.5885814333386145e-01, -4.0163200573859079e-18, -9.2840580257628812e-35, -2.8567420029274875e-51},
	{1.6188639378011183e-01, 1.1850519643573528e-17, -5.0440990519162957e-34, 3.0510028707928009e-50},
	{1.6491312048996992e-01, -7.0405288319166738e-19, 3.3211107491245527e-35, 8.6663299254686031e-52},
	{1.6793829497473117e-01, 5.4284533721558139e-18, -3.3263339336181369e-34, -1.8536367335123848e-50},
	{1.7096188876030122e-01, 9.1919980181759094e-18, -6.7688743940982606e-34, -1.0377711384318389e-50},
	{1.7398387338746382e-01, 5.8151994618107928e-18, -1.6751014298301606e-34, -6.6982259797164963e-51},
	{1.7700422041214875e-01, 6.7329300635408167e-18, 2.8042736644246623e-34, 3.6786888232793599e-51},
	{1.8002290140569951e-01, 7.9701826047392143e-18, -7.0765920110524977e-34, 1.9622512608461784e-50},
	{1.8303988795514095e-01, 7.7349918688637383e-18, -4.4803769968145083e-34, 1.1201148793328890e-50},
	{1.8605515166344666e-01, -1.2564893007679552e-17, 7.5953844248530810e-34, -3.8471695132415039e-51},
	{1.8906866414980622e-01, -7.6208955803527778e-18, -4.4792298656662981e-34, -4.4136824096645007e-50},
	{1.9208039704989244e-01, 4.3348343941174903e-18, -2.3404121848139937e-34, 1.5789970962611856e-50},
	{1.9509032201612828e-01, -7.9910790684617313e-18, 6.1846270024220713e-34, -3.5840270918032937e-50},
	{1.9809841071795359e-01, -1.8434411800689445e-18, 1.4139031318237285e-34, 1.0542811125343809e-50},
	{2.0110463484209190e-01, 1.1010032669300739e-17, -3.9123576757413791e-34, 2.4084852500063531e-51},
	{2.0410896609281687e-01, 6.0941297773957752e-18, -2.8275409970449641e-34, 4.6101008563532989e-51},
	{2.0711137619221856e-01, -1.0613362528971356e-17, 2.2456805112690884e-34, 1.3483736125280904e-50},
	{2.1011183688046961e-01, 1.1561548476512844e-17, 6.0355905610401254e-34, 3.3329909618405675e-50},
	{2.1311031991609136e-01, 1.2031873821063860e-17, -3.4142699719695635e-34, -1.2436262780241778e-50},
	{2.1610679707621952e-01, -1.0111196082609117e-17, 7.2789545335189643e-34, -2.9347540365258610e-50},
	{2.1910124015686980e-01, -3.6513812299150776e-19, -2.3359499418606442e-35, 3.1785298198458653e-52},
	{2.2209362097320354e-01, -3.0337210995812162e-18, 6.6654668033632998e-35, 2.0110862322656942e-51},
	{2.2508391135979283e-01, 3.9507040822556510e-18, 2.4287993958305375e-35, 5.6662797513020322e-52},
	{2.2807208317088573e-01, 8.2361837339258012e-18, 6.9786781316397937e-34, -6.4122962482639504e-51},
	{2.3105810828067111e-01, 1.0129787149761869e-17, -6.9359234615816044e-34, -2.8877355604883782e-50},
	{2.3404195858354343e-01, -6.9922402696101173e-18, -5.7323031922750280e-34, 5.3092579966872727e-51},
	{2.3702360599436720e-01, 8.8544852285039918e-18, 1.3588480826354134e-34, 1.0381022520213867e-50},
	{2.4000302244874150e-01, -1.2137758975632164e-17, -2.6448807731703891e-34, -1.9929733800670473e-51},
	{2.4298017990326390e-01, -8.7514315297196632e-18, -6.5723260373079431e-34, -1.0333158083172177e-50},
	{2.4595505033579462e-01, -1.1129044052741832e-17, 4.3805998202883397e-34, 1.2219399554686291e-50},
	{2.4892760574572018e-01, -8.1783436100020990e-18, 5.5666875261111840e-34, 3.8080473058748167e-50},
	{2.5189781815421697e-01, -1.7591436032517039e-17, -1.0959681232525285e-33, 5.6209426020232456e-50},
	{2.5486565960451457e-01, -1.3602299806901461e-19, -6.0073844642762535e-36, -3.0072751311893878e-52},
	{2.5783110216215899e-01, 1.8480038630879957e-17, 3.3201664714047599e-34, -5.5547819290576764e-51},
	{2.6079411791527551e-01, 4.2721420983550075e-18, 5.6782126934777920e-35, 3.1428338084365397e-51},
	{2.6375467897483140e-01, -1.8837947680038700e-17, 1.3720129045754794e-33, -8.2763406665966033e-50},
	{2.6671275747489837e-01, 2.0941222578826688e-17, -1.1303466524727989e-33, 1.9954224050508963e-50},
	{2.6966832557291509e-01, 1.5765657618133259e-17, -6.9696142173370086e-34, -4.0455346879146776e-50},
	{2.7262135544994898e-01, 7.8697166076387850e-18, 6.6179388602933372e-35, -2.7642903696386267e-51},
	{2.7557181931095814e-01, 1.9320328962556582e-17, 1.3932094180100280e-33, 1.3617253920018116e-50},
	{2.7851968938505312e-01, -1.0030273719543544e-17, 7.2592115325689254e-34, -1.0068516296655851e-50},
	{2.8146493792575800e-01, -1.2322299641274009e-17, -1.0564788706386435e-34, 7.5137424251265885e-51},
	{2.8440753721127182e-01, 2.2209268510661475e-17, -9.1823095629523708e-34, -5.2192875308892218e-50},
	{2.8734745954472951e-01, 1.5461117367645717e-17, -6.3263973663444076e-34, -2.2982538416476214e-50},
	{2.9028467725446239e-01, -1.8927978707774251e-17, 1.1522953157142315e-33, 7.4738655654716596e-50},
	{2.9321916269425863e-01, 2.2385430811901833e-17, 1.3662484646539680e-33, -4.2451325253996938e-50},
	{2.9615088824362384e-01, -2.0220736360876938e-17, -7.9252212533920413e-35, -2.8990577729572470e-51},
	{2.9907982630804048e-01, 1.6701181609219447e-18, 8.6091151117316292e-35, 3.9931286230012102e-52},
	{3.0200594931922808e-01, -1.7167666235262474e-17, 2.3336182149008069e-34, 8.3025334555220004e-51},
	{3.0492922973540243e-01, -2.2989033898191262e-17, -1.4598901099661133e-34, 3.7760487693121827e-51},
	{3.0784964004153487e-01, 2.7074088527245185e-17, 1.2568858206899284e-33, 7.2931815105901645e-50},
	{3.1076715274961147e-01, 2.0887076364048513e-17, -3.0130590791065942e-34, 1.3876739009935179e-51},
	{3.1368174039889146e-01, 1.4560447299968912e-17, 3.6564186898011595e-34, 1.1654264734999375e-50},
	{3.1659337555616585e-01, 2.1435292512726283e-17, 1.2338169231377316e-33, 3.3963542100989293e-50},
	{3.1950203081601569e-01, -1.3981562491096626e-17, 8.1730000697411350e-34, -7.7671096270210952e-50},
	{3.2240767880106985e-01, -4.0519039937959398e-18, 3.7438302780296796e-34, 8.7936731046639195e-51},
	{3.2531029216226293e-01, 7.9171249463765892e-18, -6.7576622068146391e-35, 2.3021655066929538e-51},
	{3.2820984357909255e-01, -2.6693140719641896e-17, 7.8928851447534788e-34, 2.5525163821987809e-51},
	{3.3110630575987643e-01, -2.7469465474778694e-17, -1.3401245916610206e-33, 6.5531762489976163e-50},
	{3.3399965144200938e-01, 2.2598986806288142e-17, 7.8063057192586115e-34, 2.0427600895486683e-50},
	{3.3688985339222005e-01, -4.2000940033475092e-19, -2.9178652969985438e-36, -1.1597376437036749e-52},
	{3.3977688440682685e-01, 6.6028679499418282e-18, 1.2575009988669683e-34, 2.5569067699008304e-51},
	{3.4266071731199438e-01, 1.9261518449306319e-17, -9.2754189135990867e-34, 8.5439996687390166e-50},
	{3.4554132496398904e-01, 2.7251143672916123e-17, 7.0138163601941737e-34, -1.4176292197454015e-50},
	{3.4841868024943456e-01, 3.6974420514204918e-18, 3.5532146878499996e-34, 1.9565462544501322e-50},
	{3.5129275608556715e-01, -2.2670712098795844e-17, -1.6994216673139631e-34, -1.2271556077284517e-50},
	{3.5416352542049040e-01, -1.6951763305764860e-17, 1.2772331777814617e-33, -3.3703785435843310e-50},
	{3.5703096123343003e-01, -4.8218191137919166e-19, -4.1672436994492361e-35, -7.1531167149364352e-52},
	{3.5989503653498817e-01, -1.7601687123839282e-17, 1.3375125473046791e-33, 7.9467815593584340e-50},
	{3.6275572436739723e-01, -9.1668352663749849e-18, -7.4317843956936735e-34, -2.0199582511804564e-50},
	{3.6561299780477385e-01, 1.6217898770457546e-17, 1.1286970151961055e-33, -7.1825287318139010e-50},
	{3.6846682995337232e-01, 1.0463640796159268e-17, 2.0554984738517304e-35, 1.0441861305618769e-51},
	{3.7131719395183754e-01, 3.4749239648238266e-19, -7.5151053042866671e-37, -2.8153468438650851e-53},
	{3.7416406297145799e-01, 8.0114103761962118e-18, 5.3429599813406052e-34, 1.0351378796539210e-50},
	{3.7700741021641826e-01, -2.7255302041956930e-18, 6.3646586445018137e-35, 8.3048657176503559e-52},
	{3.7984720892405116e-01, 9.9151305855172370e-18, 4.8761409697224886e-34, 1.4025084000776705e-50},
	{3.8268343236508978e-01, -1.0050772696461588e-17, -2.0605316302806695e-34, -1.2717724698085205e-50},
	{3.8551605384391885e-01, 1.5177665396472313e-17, 1.4198230518016535e-33, 5.8955167159904235e-50},
	{3.8834504669882630e-01, -1.0053770598398717e-17, 7.5942999255057131e-34, -3.1967974046654219e-50},
	{3.9117038430225387e-01, 1.7997787858243995e-17, -1.0613482402609856e-33, -5.4582148817791032e-50},
	{3.9399204006104810e-01, 9.7649241641239336e-18, -2.1233599441284617e-34, -5.5529836795340819e-51},
	{3.9680998741671031e-01, 2.0545063670840126e-17, 6.1347058801922842e-34, 1.0733788150636430e-50},
	{3.9962419984564684e-01, -1.5065497476189372e-17, -9.9653258881867298e-34, -5.7524323712725355e-50},
	{4.0243465085941843e-01, 1.0902619339328270e-17, 7.3998528125989765e-34, 2.2745784806823499e-50},
	{4.0524131400498986e-01, 9.9111401942899884e-18, -2.5169070895434648e-34, 9.2772984818436573e-53},
	{4.0804416286497869e-01, -7.0006015137351311e-18, -1.4108207334268228e-34, 1.5175546997577136e-52},
	{4.1084317105790397e-01, -2.4219835190355499e-17, -1.1418902925313314e-33, -2.0996843165093468e-50},
	{4.1363831223843456e-01, -1.0393984940597871e-17, -1.1481681174503880e-34, -2.0281052851028680e-51},
	{4.1642956009763721e-01, -2.5475580413131732e-17, -3.4482678506112824e-34, 7.1788619351865480e-51},
	{4.1921688836322396e-01, -4.2232463750110590e-18, -3.6053023045255790e-34, -2.2209673210025631e-50},
	{4.2200027079979968e-01, 4.3543266994128527e-18, 3.1734310272251190e-34, -1.3573247980738668e-50},
	{4.2477968120910881e-01, 2.7462312204277281e-17, -4.6552847802111948e-34, 6.5961781099193122e-51},
	{4.2755509343028208e-01, 9.4111898162954726e-18, -1.7446682426598801e-34, -2.2054492626480169e-51},
	{4.3032648134008261e-01, 2.2259686974092690e-17, 8.5972591314085075e-34, -2.9420897889003020e-50},
	{4.3309381885315196e-01, 1.1224283329847517e-17, 5.3223748041075651e-35, 5.3926192627014212e-51},
	{4.3585707992225547e-01, 1.6230515450644527e-17, -6.4371449063579431e-35, -6.9102436481386757e-51},
	{4.3861623853852766e-01, -2.0883315831075090e-17, -1.4259583540891877e-34, 6.3864763590657077e-52},
	{4.4137126873171667e-01, 2.2360783886964969e-17, 1.1864769603515770e-34, -3.8087003266189232e-51},
	{4.4412214457042926e-01, -2.4218874422178315e-17, 2.2205230838703907e-34, 9.2133035911356258e-51},
	{4.4686884016237421e-01, -1.9222136142309382e-17, -4.4425678589732049e-35, -1.3673609292149535e-51},
	{4.4961132965460660e-01, 4.8831924232035243e-18, 2.7151084498191381e-34, -1.5653993171613154e-50},
	{4.5234958723377089e-01, -1.4827977472196122e-17, -7.6947501088972324e-34, 1.7656856882031319e-50},
	{4.5508358712634384e-01, -1.2379906758116472e-17, 5.5289688955542643e-34, -8.5382312840209386e-51},
	{4.5781330359887723e-01, -8.4554254922295949e-18, -6.3770394246764263e-34, 3.1778253575564249e-50},
	{4.6053871095824001e-01, 1.8488777492177872e-17, -1.0527732154209725e-33, 3.3235593490947102e-50},
	{4.6325978355186020e-01, -7.3514924533231707e-18, 6.7175396881707035e-34, 3.9594127612123379e-50},
	{4.6597649576796618e-01, -3.3023547778235135e-18, 3.4904677050476886e-35, 3.4483855263874246e-51},
	{4.6868882203582796e-01, -2.2949251681845054e-17, -1.1364757641823658e-33, 6.8840522501918612e-50},
	{4.7139673682599764e-01, 6.5166781360690130e-18, 2.9457546966235984e-34, -6.2159717738836630e-51},
	{4.7410021465055002e-01, -8.1451601548978075e-18, -3.4789448555614422e-34, -1.1681943974658508e-50},
	{4.7679923006332214e-01, -1.0293515338305794e-17, -3.6582045008369952e-34, 1.7424131479176475e-50},
	{4.7949375766015301e-01, 1.8419999662684771e-17, -1.3040838621273312e-33, 1.0977131822246471e-50},
	{4.8218377207912277e-01, -2.5861500925520442e-17, -6.2913197606500007e-36, 4.0802359808684726e-52},
	{4.8486924800079112e-01, -1.8034004203262245e-17, -3.5244276906958044e-34, -1.7138318654749246e-50},
	{4.8755016014843594e-01, 1.4231090931273653e-17, -1.8277733073262697e-34, -1.5208291790429557e-51},
	{4.9022648328829116e-01, -5.1496145643440404e-18, -3.6903027405284104e-34, 1.5172940095151304e-50},
	{4.9289819222978404e-01, -1.0257831676562186e-18, 6.9520817760885069e-35, -2.4260961214090389e-51},
	{4.9556526182577254e-01, -9.4323241942365362e-18, 3.1212918657699143e-35, 4.2009072375242736e-52},
	{4.9822766697278187e-01, -1.6126383830540798e-17, -1.5092897319298871e-33, 1.1049298890895917e-50},
	{5.0088538261124083e-01, -3.9604015147074639e-17, -2.2208395201898007e-33, 1.3648202735839417e-49},
	{5.0353838372571758e-01, -1.6731308204967497e-17, -1.0140233644074786e-33, 4.0953071937671477e-50},
	{5.0618664534515534e-01, -4.8321592986493711e-17, 9.2858107226642252e-34, 4.2699802401037005e-50},
	{5.0883014254310699e-01, 4.7836968268014130e-17, -1.0727022928806035e-33, 2.7309374513672757e-50},
	{5.1146885043797041e-01, -1.3088001221007579e-17, 4.0929033363366899e-34, -3.7952190153477926e-50},
	{5.1410274419322177e-01, -4.5712707523615624e-17, 1.5488279442238283e-33, -2.5853959305521130e-50},
	{5.1673179901764987e-01, 8.3018617233836515e-18, 5.8251027467695202e-34, -2.2812397190535076e-50},
	{5.1935599016558964e-01, -5.5331248144171145e-17, -3.1628375609769026e-35, -2.4091972051188571e-51},
	{5.2197529293715439e-01, -4.6555795692088883e-17, 4.6378980936850430e-34, -3.3470542934689532e-51},
	{5.2458968267846895e-01, -4.3068869040082345e-17, -4.2013155291932055e-34, -1.5096069926700274e-50},
	{5.2719913478190139e-01, -4.2202983480560619e-17, 8.5585916184867295e-34, 7.9974339336732307e-50},
	{5.2980362468629472e-01, -4.8067841706482342e-17, 5.8309721046630296e-34, -8.9740761521756660e-51},
	{5.3240312787719801e-01, -4.1020306135800895e-17, -1.9239996374230821e-33, -1.5326987913812184e-49},
	{5.3499761988709726e-01, -5.3683132708358134e-17, -1.3900569918838112e-33, 2.7154084726474092e-50},
	{5.3758707629564551e-01, -2.2617365388403054e-17, -5.9787279033447075e-34, 3.1204419729043625e-51},
	{5.4017147272989285e-01, 2.7072447965935839e-17, 1.1698799709213829e-33, -5.9094668515881500e-50},
	{5.4275078486451589e-01, 1.7148261004757101e-17, -1.3525905925200870e-33, 4.9724411290727323e-50},
	{5.4532498842204646e-01, -4.1517817538384258e-17, -1.5318930219385941e-33, 6.3629921101413974e-50},
	{5.4789405917310019e-01, -2.4065878297113363e-17, -3.5639213669362606e-36, -2.6013270854271645e-52},
	{5.5045797293660481e-01, -8.3319903015807663e-18, -2.3058454035767633e-34, -2.1611290432369010e-50},
	{5.5301670558002758e-01, -4.7061536623798204e-17, -1.0617111545918056e-33, -1.6196316144407379e-50},
	{5.5557023301960218e-01, 4.7094109405616768e-17, -2.0640520383682921e-33, 1.2290163188567138e-49},
	{5.5811853122055610e-01, 1.3481176324765226e-17, -5.5016743873011438e-34, -2.3484822739335416e-50},
	{5.6066157619733603e-01, -7.3956418153476152e-18, 3.9680620611731193e-34, 3.1995952200836223e-50},
	{5.6319934401383409e-01, 2.3835775146854829e-17, 1.3511793173769814e-34, 9.3201311581248143e-51},
	{5.6573181078361323e-01, -3.4096079596590466e-17, -1.7073289744303546e-33, 8.9147089975404507e-50},
	{5.6825895267013160e-01, -5.0935673642769248e-17, -1.6274356351028249e-33, 9.8183151561702966e-51},
	{5.7078074588696726e-01, 2.4568151455566208e-17, -1.2844481247560350e-33, -1.8037634376936261e-50},
	{5.7329716669804220e-01, 8.5176611669306400e-18, -6.4443208788026766e-34, 2.2546105543273003e-50},
	{5.7580819141784534e-01, -3.7909495458942734e-17, -2.7433738046854309e-33, 1.1130841524216795e-49},
	{5.7831379641165559e-01, -2.6237691512372831e-17, 1.3679051680738167e-33, -3.1409808935335900e-50},
	{5.8081395809576453e-01, 1.8585338586613408e-17, 2.7673843114549181e-34, 1.9605349619836937e-50},
	{5.8330865293769829e-01, 3.4516601079044858e-18, 1.8065977478946306e-34, -6.3953958038544646e-51},
	{5.8579785745643886e-01, -3.7485501964311294e-18, 2.7965403775536614e-34, -7.1816936024157202e-51},
	{5.8828154822264533e-01, -2.9292166725006846e-17, -2.3744954603693934e-33, -1.1571631191512480e-50},
	{5.9075970185887428e-01, -4.7013584170659542e-17, 2.4808417611768356e-33, 1.2598907673643198e-50},
	{5.9323229503979980e-01, 1.2892320944189053e-17, 5.3058364776359583e-34, 4.1141674699390052e-50},
	{5.9569930449243336e-01, -1.3438641936579467e-17, -6.7877687907721049e-35, -5.6046937531684890e-51},
	{5.9816070699634227e-01, 3.8801885783000657e-17, -1.2084165858094663e-33, -4.0456610843430061e-50},
	{6.0061647938386897e-01, -4.6398198229461932e-17, -1.6673493003710801e-33, 5.1982824378491445e-50},
	{6.0306659854034816e-01, 3.7323357680559650e-17, 2.7771920866974305e-33, -1.6194229649742458e-49},
	{6.0551104140432555e-01, -3.1202672493305677e-17, 1.2761267338680916e-33, -4.0859368598379647e-50},
	{6.0794978496777363e-01, 3.5160832362096660e-17, -2.5546242776778394e-34, -1.4085313551220694e-50},
	{6.1038280627630948e-01, -2.2563265648229169e-17, 1.3185575011226730e-33, 8.2316691420063460e-50},
	{6.1281008242940971e-01, -4.2693476568409685e-18, 2.5839965886650320e-34, 1.6884412005622537e-50},
	{6.1523159058062682e-01, 2.6231417767266950e-17, -1.4095366621106716e-33, 7.2058690491304558e-50},
	{6.1764730793780398e-01, -4.7478594510902452e-17, -7.2986558263123996e-34, -3.0152327517439154e-50},
	{6.2005721176328921e-01, -2.7983410837681118e-17, 1.1649951056138923e-33, -5.4539089117135207e-50},
	{6.2246127937414997e-01, 5.2940728606573002e-18, -4.8486411215945827e-35, 1.2696527641980109e-52},
	{6.2485948814238634e-01, 3.3671846037243900e-17, -2.7846053391012096e-33, 5.6102718120012104e-50},
	{6.2725181549514408e-01, 3.0763585181253225e-17, 2.7068930273498138e-34, -1.1172240309286484e-50},
	{6.2963823891492698e-01, 4.1115334049626806e-17, -1.9167473580230747e-33, 1.1118424028161730e-49},
	{6.3201873593980906e-01, -4.0164942296463612e-17, -7.2208643641736723e-34, 3.7828920470544344e-50},
	{6.3439328416364549e-01, 1.0420901929280035e-17, 4.1174558929280492e-34, -1.4464152986630705e-51},
	{6.3676186123628420e-01, 3.1419048711901611e-17, -2.2693738415126449e-33, -1.6023584204297388e-49},
	{6.3912444486377573e-01, 1.2416796312271043e-17, -6.2095419626356605e-34, 2.7762065999506603e-50},
	{6.4148101280858316e-01, -9.9883430115943310e-18, 4.1969230376730128e-34, 5.6980543799257597e-51},
	{6.4383154288979150e-01, -3.2084798795046886e-17, -1.2595311907053305e-33, -4.0205885230841536e-50},
	{6.4617601298331639e-01, -2.9756137382280815e-17, -1.0275370077518259e-33, 8.0852478665893014e-51},
	{6.4851440102211244e-01, 3.9870270313386831e-18, 1.9408388509540788e-34, -5.1798420636193190e-51},
	{6.5084668499638088e-01, 3.9714670710500257e-17, 2.9178546787002963e-34, 3.8140635508293278e-51},
	{6.5317284295377676e-01, 8.5695642060026238e-18, -6.9165322305070633e-34, 2.3873751224185395e-50},
	{6.5549285299961535e-01, 3.5638734426385005e-17, 1.2695365790889811e-33, 4.3984952865412050e-50},
	{6.5780669329707864e-01, 1.9580943058468545e-17, -1.1944272256627192e-33, 2.8556402616436858e-50},
	{6.6011434206742048e-01, -1.3960054386823638e-19, 6.1515777931494047e-36, 5.3510498875622660e-52},
	{6.6241577759017178e-01, -2.2615508885764591e-17, 5.0177050318126862e-34, 2.9162532399530762e-50},
	{6.6471097820334490e-01, -3.6227793598034367e-17, -9.0607934765540427e-34, 3.0917036342380213e-50},
	{6.6699992230363747e-01, 3.5284364997428166e-17, -1.0382057232458238e-33, 7.3812756550167626e-50},
	{6.6928258834663612e-01, -5.4592652417447913e-17, -2.5181014709695152e-33, -1.6867875999437174e-49},
	{6.7155895484701844e-01, -4.0489037749296692e-17, 3.1995835625355681e-34, -1.4044414655670960e-50},
	{6.7382900037875604e-01, 2.3091901236161086e-17, 5.7428037192881319e-34, 1.1240668354625977e-50},
	{6.7609270357531592e-01, 3.7256902248049466e-17, 1.7059417895764375e-33, 9.7326347795300652e-50},
	{6.7835004312986147e-01, 1.8302093041863122e-17, 9.5241675746813072e-34, 5.0328101116133503e-50},
	{6.8060099779545302e-01, 2.8473293354522047e-17, 4.1331805977270903e-34, 4.2579030510748576e-50},
	{6.8284554638524808e-01, -1.2958058061524531e-17, 1.8292386959330698e-34, 3.4536209116044487e-51},
	{6.8508366777270036e-01, 2.5948135194645137e-17, -8.5030743129500702e-34, -6.9572086141009930e-50},
	{6.8731534089175916e-01, -5.5156158714917168e-17, 1.1896489854266829e-33, -7.8505896218220662e-51},
	{6.8954054473706694e-01, -1.5889323294806790e-17, 9.1242356240205712e-34, 3.8315454152267638e-50},
	{6.9175925836415775e-01, 2.7406078472410668e-17, 1.3286508943202092e-33, 1.0651869129580079e-51},
	{6.9397146088965400e-01, 7.4345076956280137e-18, 7.5061528388197460e-34, -1.5928000240686583e-50},
	{6.9617713149146299e-01, -4.1224081213582889e-17, -3.1838716762083291e-35, -3.9625587412119131e-51},
	{6.9837624940897280e-01, 4.8988282435667768e-17, 1.9134010413244152e-33, 2.6161153243793989e-50},
	{7.0056879394324834e-01, 3.1027960192992922e-17, 9.5638250509179997e-34, 4.5896916138107048e-51},
	{7.0275474445722530e-01, 2.5278294383629822e-18, -8.6985561210674942e-35, -5.6899862307812990e-51},
	{7.0493408037590488e-01, 2.7608725585748502e-17, 2.9816599471629137e-34, 1.1533044185111206e-50},
	{7.0710678118654757e-01, -4.8336466567264567e-17, 2.0693376543497068e-33, 2.4677734957341755e-50}
};
static __constant gqd_real d_cos_table_qd[256] =
{
	{9.9999529380957619E-001, -1.9668064285322189E-017, -6.3053955095883481E-034, 5.3266110855726731E-052},
	{9.9998117528260111E-001, 3.3568103522895585E-017, -1.4740132559368063E-035, 9.8603097594755596E-052},
	{9.9995764455196390E-001, -3.1527836866647287E-017, 2.6363251186638437E-033, 1.0007504815488399E-049},
	{9.9992470183914450E-001, 3.7931082512668012E-017, -8.5099918660501484E-035, -4.9956973223295153E-051},
	{9.9988234745421256E-001, -3.5477814872408538E-017, 1.7102001035303974E-033, -1.0725388519026542E-049},
	{9.9983058179582340E-001, 1.8825140517551119E-017, -5.1383513457616937E-034, -3.8378827995403787E-050},
	{9.9976940535121528E-001, 4.2681177032289012E-017, 1.9062302359737099E-033, -6.0221153262881160E-050},
	{9.9969881869620425E-001, -2.9851486403799753E-017, -1.9084787370733737E-033, 5.5980260344029202E-051},
	{9.9961882249517864E-001, -4.1181965521424734E-017, 2.0915365593699916E-033, 8.1403390920903734E-050},
	{9.9952941750109314E-001, 2.0517917823755591E-017, -4.7673802585706520E-034, -2.9443604198656772E-050},
	{9.9943060455546173E-001, 3.9644497752257798E-017, -2.3757223716722428E-034, -1.2856759011361726E-051},
	{9.9932238458834954E-001, -4.2858538440845682E-017, 3.3235101605146565E-034, -8.3554272377057543E-051},
	{9.9920475861836389E-001, 9.1796317110385693E-018, 5.5416208185868570E-034, 8.0267046717615311E-052},
	{9.9907772775264536E-001, 2.1419007653587032E-017, -7.9048203318529618E-034, -5.3166296181112712E-050},
	{9.9894129318685687E-001, -2.0610641910058638E-017, -1.2546525485913485E-033, -7.5175888806157064E-050},
	{9.9879545620517241E-001, -1.2291693337075465E-017, 2.4468446786491271E-034, 1.0723891085210268E-050},
	{9.9864021818026527E-001, -4.8690254312923302E-017, -2.9470881967909147E-034, -1.3000650761346907E-050},
	{9.9847558057329477E-001, -2.2002931182778795E-017, -1.2371509454944992E-033, -2.4911225131232065E-050},
	{9.9830154493389289E-001, -5.1869402702792278E-017, 1.0480195493633452E-033, -2.8995649143155511E-050},
	{9.9811811290014918E-001, 2.7935487558113833E-017, 2.4423341255830345E-033, -6.7646699175334417E-050},
	{9.9792528619859600E-001, 1.7143659778886362E-017, 5.7885840902887460E-034, -9.2601432603894597E-051},
	{9.9772306664419164E-001, -2.6394475274898721E-017, -1.6176223087661783E-034, -9.9924942889362281E-051},
	{9.9751145614030345E-001, 5.6007205919806937E-018, -5.9477673514685690E-035, -1.4166807162743627E-054},
	{9.9729045667869021E-001, 9.1647695371101735E-018, 6.7824134309739296E-034, -8.6191392795543357E-052},
	{9.9706007033948296E-001, 1.6734093546241963E-017, -1.3169951440780028E-033, 1.0311048767952477E-050},
	{9.9682029929116567E-001, 4.7062820708615655E-017, 2.8412041076474937E-033, -8.0006155670263622E-050},
	{9.9657114579055484E-001, 1.1707179088390986E-017, -7.5934413263024663E-034, 2.8474848436926008E-050},
	{9.9631261218277800E-001, 1.1336497891624735E-017, 3.4002458674414360E-034, 7.7419075921544901E-052},
	{9.9604470090125197E-001, 2.2870031707670695E-017, -3.9184839405013148E-034, -3.7081260416246375E-050},
	{9.9576741446765982E-001, -2.3151908323094359E-017, -1.6306512931944591E-034, -1.5925420783863192E-051},
	{9.9548075549192694E-001, 3.2084621412226554E-018, -4.9501292146013023E-036, -2.7811428850878516E-052},
	{9.9518472667219693E-001, -4.2486913678304410E-017, 1.3315510772504614E-033, 6.7927987417051888E-050},
	{9.9487933079480562E-001, 4.2130813284943662E-018, -4.2062597488288452E-035, 2.5157064556087620E-051},
	{9.9456457073425542E-001, 3.6745069641528058E-017, -3.0603304105471010E-033, 1.0397872280487526E-049},
	{9.9424044945318790E-001, 4.4129423472462673E-017, -3.0107231708238066E-033, 7.4201582906861892E-050},
	{9.9390697000235606E-001, -1.8964849471123746E-017, -1.5980853777937752E-035, -8.5374807150597082E-052},
	{9.9356413552059530E-001, 2.9752309927797428E-017, -4.5066707331134233E-034, -3.3548191633805036E-050},
	{9.9321194923479450E-001, 3.3096906261272262E-017, 1.5592811973249567E-033, 1.4373977733253592E-050},
	{9.9285041445986510E-001, -1.4094517733693302E-017, -1.1954558131616916E-033, 1.8761873742867983E-050},
	{9.9247953459870997E-001, 3.1093055095428906E-017, -1.8379594757818019E-033, -3.9885758559381314E-051},
	{9.9209931314219180E-001, -3.9431926149588778E-017, -6.2758062911047230E-034, -1.2960929559212390E-050},
	{9.9170975366909953E-001, -2.3372891311883661E-018, 2.7073298824968591E-035, -1.2569459441802872E-051},
	{9.9131085984611544E-001, -2.5192111583372105E-017, -1.2852471567380887E-033, 5.2385212584310483E-050},
	{9.9090263542778001E-001, 1.5394565094566704E-017, -1.0799984133184567E-033, 2.7451115960133595E-051},
	{9.9048508425645709E-001, -5.5411437553780867E-017, -1.4614017210753585E-033, -3.8339374397387620E-050},
	{9.9005821026229712E-001, -1.7055485906233963E-017, 1.3454939685758777E-033, 7.3117589137300036E-050},
	{9.8962201746320089E-001, -5.2398217968132530E-017, 1.3463189211456219E-033, 5.8021640554894872E-050},
	{9.8917650996478101E-001, -4.0987309937047111E-017, -4.4857560552048437E-034, -3.9414504502871125E-050},
	{9.8872169196032378E-001, -1.0976227206656125E-017, 3.2311342577653764E-034, 9.6367946583575041E-051},
	{9.8825756773074946E-001, 2.7030607784372632E-017, 7.7514866488601377E-035, 2.1019644956864938E-051},
	{9.8778414164457218E-001, -2.3600693397159021E-017, -1.2323283769707861E-033, 3.0130900716803339E-050},
	{9.8730141815785843E-001, -5.2332261255715652E-017, -2.7937644333152473E-033, 1.2074160567958408E-049},
	{9.8680940181418553E-001, -5.0287214351061075E-017, -2.2681526238144461E-033, 4.4003694320169133E-050},
	{9.8630809724459867E-001, -2.1520877103013341E-017, 1.1866528054187716E-033, -7.8532199199813836E-050},
	{9.8579750916756748E-001, -5.1439452979953012E-017, 2.6276439309996725E-033, 7.5423552783286347E-050},
	{9.8527764238894122E-001, 2.3155637027900207E-017, -7.5275971545764833E-034, 1.0582231660456094E-050},
	{9.8474850180190421E-001, 1.0548144061829957E-017, 2.8786145266267306E-034, -3.6782210081466112E-051},
	{9.8421009238692903E-001, 4.7983922627050691E-017, 2.2597419645070588E-034, 1.7573875814863400E-050},
	{9.8366241921173025E-001, 1.9864948201635255E-017, -1.0743046281211033E-035, 1.7975662796558100E-052},
	{9.8310548743121629E-001, 4.2170007522888628E-017, 8.2396265656440904E-034, -8.0803700139096561E-050},
	{9.8253930228744124E-001, 1.5149580813777224E-017, -4.1802771422186237E-034, -2.2150174326226160E-050},
	{9.8196386910955524E-001, 2.1108443711513084E-017, -1.5253013442896054E-033, -6.8388082079337969E-050},
	{9.8137919331375456E-001, 1.3428163260355633E-017, -6.5294290469962986E-034, 2.7965412287456268E-051},
	{9.8078528040323043E-001, 1.8546939997825006E-017, -1.0696564445530757E-033, 6.6668174475264961E-050},
	{9.8018213596811743E-001, -3.6801786963856159E-017, 6.3245171387992842E-034, 1.8600176137175971E-050},
	{9.7956976568544052E-001, 1.5573991584990420E-017, -1.3401066029782990E-033, -1.7263702199862149E-050},
	{9.7894817531906220E-001, -2.3817727961148053E-018, -1.0694750370381661E-034, -8.2293047196087462E-051},
	{9.7831737071962765E-001, -2.1623082233344895E-017, 1.0970403012028032E-033, 7.7091923099369339E-050},
	{9.7767735782450993E-001, 5.0514136167059628E-017, -1.3254751701428788E-033, 7.0161254312124538E-050},
	{9.7702814265775439E-001, -4.3353875751555997E-017, 5.4948839831535478E-034, -9.2755263105377306E-051},
	{9.7636973133002114E-001, 9.3093931526213780E-018, -4.1184949155685665E-034, -3.1913926031393690E-050},
	{9.7570213003852857E-001, -2.5572556081259686E-017, -9.3174244508942223E-034, -8.3675863211646863E-051},
	{9.7502534506699412E-001, 2.6642660651899135E-017, 1.7819392739353853E-034, -3.3159625385648947E-051},
	{9.7433938278557586E-001, 2.3041221476151512E-018, 1.0758686005031430E-034, 5.1074116432809478E-051},
	{9.7364424965081198E-001, -5.1729808691005871E-017, -1.5508473005989887E-033, -1.6505125917675401E-049},
	{9.7293995220556018E-001, -3.1311211122281800E-017, -2.6874087789006141E-033, -2.1652434818822145E-051},
	{9.7222649707893627E-001, 3.6461169785938221E-017, 3.0309636883883133E-033, -1.2702716907967306E-051},
	{9.7150389098625178E-001, -7.9865421122289046E-018, -4.3628417211263380E-034, 3.4307517798759352E-051},
	{9.7077214072895035E-001, -4.7992163325114922E-017, 3.0347528910975783E-033, 8.5989199506479701E-050},
	{9.7003125319454397E-001, 1.8365300348428844E-017, -1.4311097571944918E-033, 8.5846781998740697E-051},
	{9.6928123535654853E-001, -4.5663660261927896E-017, 9.6147526917239387E-034, 8.1267605207871330E-051},
	{9.6852209427441727E-001, 4.9475074918244771E-017, 2.8558738351911241E-033, 6.2948422316507461E-050},
	{9.6775383709347551E-001, -4.5512132825515820E-017, -1.4127617988719093E-033, -8.4620609089704578E-050},
	{9.6697647104485207E-001, 3.8496228837337864E-017, -5.3881631542745647E-034, -3.5221863171458959E-050},
	{9.6619000344541250E-001, 5.1298840401665493E-017, 1.4564075904769808E-034, 1.0095973971377432E-050},
	{9.6539444169768940E-001, -2.3745389918392156E-017, 5.9221515590053862E-034, -3.8811192556231094E-050},
	{9.6458979328981276E-001, -3.4189470735959786E-017, 2.2982074155463522E-033, -4.5128791045607634E-050},
	{9.6377606579543984E-001, 2.6463950561220029E-017, -2.9073234590199323E-036, -1.2938328629395601E-052},
	{9.6295326687368388E-001, 8.9341960404313634E-018, -3.9071244661020126E-034, 1.6212091116847394E-050},
	{9.6212140426904158E-001, 1.5236770453846305E-017, -1.3050173525597142E-033, 7.9016122394092666E-050},
	{9.6128048581132064E-001, 2.0933955216674039E-018, 1.0768607469015692E-034, -5.9453639304361774E-051},
	{9.6043051941556579E-001, 2.4653904815317185E-017, -1.3792169410906322E-033, -4.7726598378506903E-051},
	{9.5957151308198452E-001, 1.1000640085000957E-017, -4.2036030828223975E-034, 4.0023704842606573E-051},
	{9.5870347489587160E-001, -4.3685014392372053E-017, 2.2001800662729131E-033, -1.0553721324358075E-049},
	{9.5782641302753291E-001, -1.7696710075371263E-017, 1.9164034110382190E-034, 8.1489235071754813E-051},
	{9.5694033573220882E-001, 4.0553869861875701E-017, -1.7147013364302149E-033, 2.5736745295329455E-050},
	{9.5604525134999641E-001, 3.7705045279589067E-017, 1.9678699997347571E-033, 8.5093177731230180E-050},
	{9.5514116830577067E-001, 5.0088652955014668E-017, -2.6983181838059211E-033, 1.0102323575596493E-049},
	{9.5422809510910567E-001, -3.7545901690626874E-017, 1.4951619241257764E-033, -8.2717333151394973E-050},
	{9.5330604035419386E-001, -2.5190738779919934E-017, -1.4272239821134379E-033, -4.6717286809283155E-050},
	{9.5237501271976588E-001, -2.0269300462299272E-017, -1.0635956887246246E-033, -3.5514537666487619E-050},
	{9.5143502096900834E-001, 3.1350584123266695E-017, -2.4824833452737813E-033, 9.5450335525380613E-051},
	{9.5048607394948170E-001, 1.9410097562630436E-017, -8.1559393949816789E-034, -1.0501209720164562E-050},
	{9.4952818059303667E-001, -7.5544151928043298E-018, -5.1260245024046686E-034, 1.8093643389040406E-050},
	{9.4856134991573027E-001, 2.0668262262333232E-017, -5.9440730243667306E-034, 1.4268853111554300E-050},
	{9.4758559101774109E-001, 4.3417993852125991E-017, -2.7728667889840373E-034, 5.5709160196519968E-051},
	{9.4660091308328353E-001, 3.5056800210680730E-017, 9.8578536940318117E-034, 6.6035911064585197E-050},
	{9.4560732538052128E-001, 4.6019102478523738E-017, -6.2534384769452059E-034, 1.5758941215779961E-050},
	{9.4460483726148026E-001, 8.8100545476641165E-018, 5.2291695602757842E-034, -3.3487256018407123E-050},
	{9.4359345816196039E-001, -2.4093127844404214E-017, 1.0283279856803939E-034, -2.3398232614531355E-051},
	{9.4257319760144687E-001, 1.3235564806436886E-017, -5.7048262885386911E-035, 3.9947050442753744E-051},
	{9.4154406518302081E-001, -2.7896379547698341E-017, 1.6273236356733898E-033, -5.3075944708471203E-051},
	{9.4050607059326830E-001, 2.8610421567116268E-017, 2.9261501147538827E-033, -2.6849867690896925E-050},
	{9.3945922360218992E-001, -7.0152867943098655E-018, -5.6395693818011210E-034, 3.5568142678987651E-050},
	{9.3840353406310806E-001, 5.4242545044795490E-017, -1.9039966607859759E-033, -1.5627792988341215E-049},
	{9.3733901191257496E-001, -3.6570926284362776E-017, -1.1902940071273247E-033, -1.1215082331583223E-050},
	{9.3626566717027826E-001, -1.3013766145497654E-017, 5.2229870061990595E-034, -3.3972777075634108E-051},
	{9.3518350993894761E-001, -3.2609395302485065E-017, -8.1813015218875245E-034, 5.5642140024928139E-050},
	{9.3409255040425887E-001, 4.4662824360767511E-017, -2.5903243047396916E-033, 8.1505209004343043E-050},
	{9.3299279883473885E-001, 4.2041415555384355E-017, 9.0285896495521276E-034, 5.3019984977661259E-050},
	{9.3188426558166815E-001, -4.0785944377318095E-017, 1.7631450298754169E-033, 2.5776403305507453E-050},
	{9.3076696107898371E-001, 1.9703775102838329E-017, 6.5657908718278205E-034, -1.9480347966259524E-051},
	{9.2964089584318121E-001, 5.1282530016864107E-017, 2.3719739891916261E-034, -1.7230065426917127E-050},
	{9.2850608047321559E-001, -2.3306639848485943E-017, -7.7799084333208503E-034, -5.8597558009300305E-050},
	{9.2736252565040111E-001, -2.7677111692155437E-017, 2.2110293450199576E-034, 2.0349190819680613E-050},
	{9.2621024213831138E-001, -3.7303754586099054E-017, 2.0464457809993405E-033, 1.3831799631231817E-049},
	{9.2504924078267758E-001, 6.0529447412576159E-018, -8.8256517760278541E-035, 1.8285462122388328E-051},
	{9.2387953251128674E-001, 1.7645047084336677E-017, -5.0442537321586818E-034, -4.0478677716823890E-050},
	{9.2270112833387852E-001, 5.2963798918539814E-017, -5.7135699628876685E-034, 3.0163671797219087E-050},
	{9.2151403934204190E-001, 4.1639843390684644E-017, 1.1891485604702356E-033, 2.0862437594380324E-050},
	{9.2031827670911059E-001, -2.7806888779036837E-017, 2.7011013677071274E-033, 1.1998578792455499E-049},
	{9.1911385169005777E-001, -2.6496484622344718E-017, 6.5403604763461920E-034, -2.8997180201186078E-050},
	{9.1790077562139050E-001, -3.9074579680849515E-017, 2.3004636541490264E-033, 3.9851762744443107E-050},
	{9.1667905992104270E-001, -4.1733978698287568E-017, 1.2094444804381172E-033, 4.9356916826097816E-050},
	{9.1544871608826783E-001, -1.3591056692900894E-017, 5.9923027475594735E-034, 2.1403295925962879E-050},
	{9.1420975570353069E-001, -3.6316182527814423E-017, -1.9438819777122554E-033, 2.8340679287728316E-050},
	{9.1296219042839821E-001, -4.7932505228039469E-017, -1.7753551889428638E-033, 4.0607782903868160E-051},
	{9.1170603200542988E-001, -2.6913273175034130E-017, -5.1928101916162528E-035, 1.1338175936090630E-051},
	{9.1044129225806725E-001, -5.0433041673313820E-017, 1.0938746257404305E-033, 9.5378272084170731E-051},
	{9.0916798309052238E-001, -3.6878564091359894E-018, 2.9951330310507693E-034, -1.2225666136919926E-050},
	{9.0788611648766626E-001, -4.9459964301225840E-017, -1.6599682707075313E-033, -5.1925202712634716E-050},
	{9.0659570451491533E-001, 3.0506718955442023E-017, -1.4478836557141204E-033, 1.8906373784448725E-050},
	{9.0529675931811882E-001, -4.1153099826889901E-017, 2.9859368705184223E-033, 5.1145293917439211E-050},
	{9.0398929312344334E-001, -6.6097544687484308E-018, 1.2728013034680357E-034, -4.3026097234014823E-051},
	{9.0267331823725883E-001, -1.9250787033961483E-017, 1.3242128993244527E-033, -5.2971030688703665E-050},
	{9.0134884704602203E-001, -1.3524789367698682E-017, 6.3605353115880091E-034, 3.6227400654573828E-050},
	{9.0001589201616028E-001, -5.0639618050802273E-017, 1.0783525384031576E-033, 2.8130016326515111E-050},
	{8.9867446569395382E-001, 2.6316906461033013E-017, 3.7003137047796840E-035, -2.3447719900465938E-051},
	{8.9732458070541832E-001, -3.6396283314867290E-017, -2.3611649895474815E-033, 1.1837247047900082E-049},
	{8.9596624975618511E-001, 4.9025099114811813E-017, -1.9440489814795326E-033, -1.7070486667767033E-049},
	{8.9459948563138270E-001, -1.7516226396814919E-017, -1.3200670047246923E-033, -1.5953009884324695E-050},
	{8.9322430119551532E-001, -4.1161239151908913E-018, 2.5380253805715999E-034, 4.2849455510516192E-051},
	{8.9184070939234272E-001, 4.6690228137124547E-018, 1.6150254286841982E-034, -3.9617448820725012E-051},
	{8.9044872324475788E-001, 1.1781931459051803E-017, -1.3346142209571930E-034, -9.4982373530733431E-051},
	{8.8904835585466457E-001, -1.1164514966766675E-017, -3.4797636107798736E-034, -1.5605079997040631E-050},
	{8.8763962040285393E-001, 1.2805091918587960E-017, 3.9948742059584459E-035, 3.8940716325338136E-051},
	{8.8622253014888064E-001, -6.7307369600274315E-018, 1.2385593432917413E-034, 2.0364014759133320E-051},
	{8.8479709843093779E-001, -9.4331469628972690E-018, -5.7106541478701439E-034, 1.8260134111907397E-050},
	{8.8336333866573158E-001, 1.5822643380255127E-017, -7.8921320007588250E-034, -1.4782321016179836E-050},
	{8.8192126434835505E-001, -1.9843248405890562E-017, -7.0412114007673834E-034, -1.0636770169389104E-050},
	{8.8047088905216075E-001, 1.6311096602996350E-017, -5.7541360594724172E-034, -4.0128611862170021E-050},
	{8.7901222642863353E-001, -4.7356837291118011E-017, 1.4388771297975192E-033, -2.9085554304479134E-050},
	{8.7754529020726124E-001, 5.0113311846499550E-017, 2.8382769008739543E-034, 1.5550640393164140E-050},
	{8.7607009419540660E-001, 5.8729024235147677E-018, 2.7941144391738458E-034, -1.8536073846509828E-050},
	{8.7458665227817611E-001, -5.7216617730397065E-019, -2.9705811503689596E-035, 8.7389593969796752E-052},
	{8.7309497841829009E-001, 7.8424672990129903E-018, -4.8685015839797165E-034, -2.2815570587477527E-050},
	{8.7159508665595109E-001, -5.5272998038551050E-017, -2.2104090204984907E-033, -9.7749763187643172E-050},
	{8.7008699110871146E-001, -4.1888510868549968E-017, 7.0900185861878415E-034, 3.7600251115157260E-050},
	{8.6857070597134090E-001, 2.7192781689782903E-019, -1.6710140396932428E-035, -1.2625514734637969E-051},
	{8.6704624551569265E-001, 3.0267859550930567E-018, -1.1559438782171572E-034, -5.3580556397808012E-052},
	{8.6551362409056909E-001, -6.3723113549628899E-018, 2.3725520321746832E-034, 1.5911880348395175E-050},
	{8.6397285612158670E-001, 4.1486355957361607E-017, 2.2709976932210266E-033, -8.1228385659479984E-050},
	{8.6242395611104050E-001, 3.7008992527383130E-017, 5.2128411542701573E-034, 2.6945600081026861E-050},
	{8.6086693863776731E-001, -3.0050048898573656E-017, -8.8706183090892111E-034, 1.5005320558097301E-050},
	{8.5930181835700836E-001, 4.2435655816850687E-017, 7.6181814059912025E-034, -3.9592127850658708E-050},
	{8.5772861000027212E-001, -4.8183447936336620E-017, -1.1044130517687532E-033, -8.7400233444645562E-050},
	{8.5614732837519447E-001, 9.1806925616606261E-018, 5.6328649785951470E-034, 2.3326646113217378E-051},
	{8.5455798836540053E-001, -1.2991124236396092E-017, 1.2893407722948080E-034, -3.6506925747583053E-052},
	{8.5296060493036363E-001, 2.7152984251981370E-017, 7.4336483283120719E-034, 4.2162417622350668E-050},
	{8.5135519310526520E-001, -5.3279874446016209E-017, 2.2281156380919942E-033, -4.0281886404138477E-050},
	{8.4974176800085244E-001, 5.1812347659974015E-017, 3.0810626087331275E-033, -2.5931308201994965E-050},
	{8.4812034480329723E-001, 1.8762563415239981E-017, 1.4048773307919617E-033, -2.4915221509958691E-050},
	{8.4649093877405213E-001, -4.7969419958569345E-017, -2.7518267097886703E-033, -7.3518959727313350E-050},
	{8.4485356524970712E-001, -4.3631360296879637E-017, -2.0307726853367547E-033, 4.3097229819851761E-050},
	{8.4320823964184544E-001, 9.6536707005959077E-019, 2.8995142431556364E-036, 9.6715076811480284E-053},
	{8.4155497743689844E-001, -3.4095465391321557E-017, -8.4130208607579595E-034, -4.9447283960568686E-050},
	{8.3989379419599952E-001, -1.6673694881511411E-017, -1.4759184141750289E-033, -7.5795098161914058E-050},
	{8.3822470555483808E-001, -3.5560085052855026E-017, 1.1689791577022643E-033, -5.8627347359723411E-050},
	{8.3654772722351201E-001, -2.0899059027066533E-017, -9.8104097821002585E-035, -3.1609177868229853E-051},
	{8.3486287498638001E-001, 4.6048430609159657E-017, -5.1827423265239912E-034, -7.0505343435504109E-051},
	{8.3317016470191319E-001, 1.3275129507229764E-018, 4.8589164115370863E-035, 4.5422281300506859E-051},
	{8.3146961230254524E-001, 1.4073856984728024E-018, 4.6951315383980830E-035, 5.1431906049905658E-051},
	{8.2976123379452305E-001, -2.9349109376485597E-018, 1.1496917934149818E-034, 3.5186665544980233E-051},
	{8.2804504525775580E-001, -4.4196593225871532E-017, 2.7967864855211251E-033, 1.0030777287393502E-049},
	{8.2632106284566353E-001, -5.3957485453612902E-017, 6.8976896130138550E-034, 3.8106164274199196E-050},
	{8.2458930278502529E-001, -2.6512360488868275E-017, 1.6916964350914386E-034, 6.7693974813562649E-051},
	{8.2284978137582632E-001, 1.5193019034505495E-017, 9.6890547246521685E-034, 5.6994562923653264E-050},
	{8.2110251499110465E-001, 3.0715131609697682E-017, -1.7037168325855879E-033, -1.1149862443283853E-049},
	{8.1934752007679701E-001, -4.8200736995191133E-017, -1.5574489646672781E-035, -9.5647853614522216E-053},
	{8.1758481315158371E-001, -1.4883149812426772E-017, -7.8273262771298917E-034, 4.1332149161031594E-050},
	{8.1581441080673378E-001, 8.2652693782130871E-018, -2.3028778135179471E-034, 1.5102071387249843E-050},
	{8.1403632970594841E-001, -5.2127351877042624E-017, -1.9047670611316360E-033, -1.6937269585941507E-049},
	{8.1225058658520388E-001, 3.1054545609214803E-017, 2.2649541922707251E-034, -7.4221684154649405E-051},
	{8.1045719825259477E-001, 2.3520367349840499E-017, -7.7530070904846341E-034, -7.2792616357197140E-050},
	{8.0865618158817498E-001, 9.3251597879721674E-018, -7.1823301933068394E-034, 2.3925440846132106E-050},
	{8.0684755354379922E-001, 4.9220603766095546E-017, 2.9796016899903487E-033, 1.5220754223615788E-049},
	{8.0503133114296355E-001, 5.1368289568212149E-017, 6.3082807402256524E-034, 7.3277646085129827E-051},
	{8.0320753148064494E-001, -3.3060609804814910E-017, -1.2242726252420433E-033, 2.8413673268630117E-050},
	{8.0137617172314024E-001, -2.0958013413495834E-017, -4.3798162198006931E-034, 2.0235690497752515E-050},
	{7.9953726910790501E-001, 2.0356723822005431E-017, -9.7448513696896360E-034, 5.3608109599696008E-052},
	{7.9769084094339116E-001, -4.6730759884788944E-017, 2.3075897077191757E-033, 3.1605567774640253E-051},
	{7.9583690460888357E-001, -3.0062724851910721E-017, -2.2496210832042235E-033, -6.5881774117183040E-050},
	{7.9397547755433717E-001, -7.4194631759921416E-018, 2.4124341304631069E-034, -4.9956808616244972E-051},
	{7.9210657730021239E-001, -3.7087850202326467E-017, -1.4874457267228264E-033, 2.9323097289153505E-050},
	{7.9023022143731003E-001, 2.3056905954954492E-017, 1.4481080533260193E-033, -7.6725237057203488E-050},
	{7.8834642762660623E-001, 3.4396993154059708E-017, 1.7710623746737170E-033, 1.7084159098417402E-049},
	{7.8645521359908577E-001, -9.7841429939305265E-018, 3.3906063272445472E-034, 5.7269505320382577E-051},
	{7.8455659715557524E-001, -8.5627965423173476E-018, -2.1106834459001849E-034, -1.6890322182469603E-050},
	{7.8265059616657573E-001, 9.0745866975808825E-018, 6.7623847404278666E-034, -1.7173237731987271E-050},
	{7.8073722857209449E-001, -9.9198782066678806E-018, -2.1265794012162715E-036, 3.0772165598957647E-054},
	{7.7881651238147598E-001, -2.4891385579973807E-017, 6.7665497024807980E-035, -6.5218594281701332E-052},
	{7.7688846567323244E-001, 7.7418602570672864E-018, -5.9986517872157897E-034, 3.0566548232958972E-050},
	{7.7495310659487393E-001, -5.2209083189826433E-017, -9.6653593393686612E-034, 3.7027750076562569E-050},
	{7.7301045336273699E-001, -3.2565907033649772E-017, 1.3860807251523929E-033, -3.9971329917586022E-050},
	{7.7106052426181382E-001, -4.4558442347769265E-017, -2.9863565614083783E-033, -6.8795262083596236E-050},
	{7.6910333764557959E-001, 5.1546455184564817E-017, 2.6142829553524292E-033, -1.6199023632773298E-049},
	{7.6713891193582040E-001, -1.8885903683750782E-017, -1.3659359331495433E-033, -2.2538834962921934E-050},
	{7.6516726562245896E-001, -3.2707225612534598E-017, 1.1177117747079528E-033, -3.7005182280175715E-050},
	{7.6318841726338127E-001, 2.6314748416750748E-018, 1.4048039063095910E-034, 8.9601886626630321E-052},
	{7.6120238548426178E-001, 3.5315510881690551E-017, 1.2833566381864357E-033, 8.6221435180890613E-050},
	{7.5920918897838807E-001, -3.8558842175523123E-017, 2.9720241208332759E-034, -1.2521388928220163E-050},
	{7.5720884650648457E-001, -1.9909098777335502E-017, 3.9409283266158482E-034, 2.0744254207802976E-050},
	{7.5520137689653655E-001, -1.9402238001823017E-017, -3.7756206444727573E-034, -2.1212242308178287E-050},
	{7.5318679904361252E-001, -3.7937789838736540E-017, -6.7009539920231559E-034, -6.7128562115050214E-051},
	{7.5116513190968637E-001, 4.3499761158645868E-017, 2.5227718971102212E-033, -6.5969709212757102E-050},
	{7.4913639452345937E-001, -4.4729078447011889E-017, -2.4206025249983768E-033, 1.1336681351116422E-049},
	{7.4710060598018013E-001, 1.1874824875965430E-017, 2.1992523849833518E-034, 1.1025018564644483E-050},
	{7.4505778544146595E-001, 1.5078686911877863E-017, 8.0898987212942471E-034, 8.2677958765323532E-050},
	{7.4300795213512172E-001, -2.5144629669719265E-017, 7.1128989512526157E-034, 3.0181629077821220E-050},
	{7.4095112535495911E-001, -1.4708616952297345E-017, -4.9550433827142032E-034, 3.1434132533735671E-050},
	{7.3888732446061511E-001, 3.4324874808225091E-017, -1.3706639444717610E-033, -3.3520827530718938E-051},
	{7.3681656887736990E-001, -2.8932468101656295E-017, -3.4649887126202378E-034, -1.8484474476291476E-050},
	{7.3473887809596350E-001, -3.4507595976263941E-017, -2.3718000676666409E-033, -3.9696090387165402E-050},
	{7.3265427167241282E-001, 1.8918673481573520E-017, -1.5123719544119886E-033, -9.7922152011625728E-051},
	{7.3056276922782759E-001, -2.9689959904476928E-017, -1.1276871244239744E-033, -3.0531520961539007E-050},
	{7.2846439044822520E-001, 1.1924642323370718E-019, 5.9001892316611011E-036, 1.2178089069502704E-052},
	{7.2635915508434601E-001, -3.1917502443460542E-017, 7.7047912412039396E-034, 4.1455880160182123E-050},
	{7.2424708295146689E-001, 2.9198471334403004E-017, 2.3027324968739464E-033, -1.2928820533892183E-051},
	{7.2212819392921535E-001, -2.3871262053452047E-017, 1.0636125432862273E-033, -4.4598638837802517E-050},
	{7.2000250796138165E-001, -2.5689658854462333E-017, -9.1492566948567925E-034, 4.4403780801267786E-050},
	{7.1787004505573171E-001, 2.7006476062511453E-017, -2.2854956580215348E-034, 9.1726903890287867E-051},
	{7.1573082528381871E-001, -5.1581018476410262E-017, -1.3736271349300259E-034, -1.2734611344111297E-050},
	{7.1358486878079364E-001, -4.2342504403133584E-017, -4.2690366101617268E-034, -2.6352370883066522E-050},
	{7.1143219574521643E-001, 7.9643298613856813E-018, 2.9488239510721469E-034, 1.6985236437666356E-050},
	{7.0927282643886569E-001, -3.7597359110245730E-017, 1.0613125954645119E-034, 8.9465480185486032E-051},
	{7.0710678118654757E-001, -4.8336466567264567E-017, 2.0693376543497068E-033, 2.4677734957341755E-050},
};

__attribute__((overloadable))
gqd_real sqrt(const gqd_real a) {
	 if (is_zero(a))
		  return Zero;

	 //!!!!!!!!!!
	 if (is_negative(a)) {
		  //TO DO: should return an error
		  //return _nan;
		  return Zero;
	 }

	 gqd_real r = make_qd((1.0 / sqrt(a.x)));
	 gqd_real h = mul_pwr2(a, 0.5);

	 r = HP(r + ((0.5 - h * sqr(r)) * r));
	 r = HP(r + ((0.5 - h * sqr(r)) * r));
	 r = HP(r + ((0.5 - h * sqr(r)) * r));

	 r = HP(r * a);

	 return r;
}

void sincos_taylor(const gqd_real a, gqd_real *sin_a, gqd_real *cos_a)
{
	const double thresh = 0.5 * _qd_eps * fabs(to_double(a));
	gqd_real p, s, t, x;

	//not use(we are NOT on CPU)
	/*if (is_zero(a))
	{
		*sin_a = Zero;
		*cos_a = One;
		 return;
	}*/

	 //x = -sqr(a);
	 x = negative(sqr(a));
	 s = a;
	 p = a;
	 int i = 0;
	 do {
		  p = HP(p * x);
		  t = HP(p * inv_fact[i]);
		  s = HP(s + t);
		  i = i + 2;
	 } while (i < n_inv_fact && fabs(to_double(t)) > thresh);

	 *sin_a = s;
	 *cos_a = sqrt(HP(1.0 - sqr(s)));
}

gqd_real sin_taylor(const gqd_real a) {
	 const double thresh = 0.5 * _qd_eps * fabs(to_double(a));
	 gqd_real p, s, t, x;

	//not use(we are NOT on CPU) if (is_zero(a)) return Zero;

	 //x = -sqr(a);
	 x = negative(sqr(a));
	 s = a;
	 p = a;
	 int i = 0;
	 do {
		  p = HP(p * x);
		  t = HP(p * inv_fact[i]);
		  s = HP(s + t);
		  i += 2;
	 } while (i < n_inv_fact && fabs(to_double(t)) > thresh);

	 return s;
}

gqd_real cos_taylor(const gqd_real a) {
	 const double thresh = 0.5 * _qd_eps;
	 gqd_real p, s, t, x;

	//not use(we are NOT on CPU) if (is_zero(a)) return One;

	 x = negative(sqr(a));
	 s = HP(1.0 + mul_pwr2(x, 0.5));
	 p = x;
	 int i = 1;
	 do {
		  p = HP(p * x);
		  t = HP(p * inv_fact[i]);
		  s = HP(s + t);
		  i += 2;
	 } while (i < n_inv_fact && fabs(to_double(t)) > thresh);

	 return s;
}

__attribute__((overloadable))
gqd_real sin(const gqd_real a)
{
	 gqd_real z, r;
	//not use(we are NOT on CPU) if (is_zero(a)) return Zero;

	 // approximately reduce modulo 2*pi
	 z = nint(HP(a / _qd_2pi));
	 r = HP(a - _qd_2pi * z);

	 // approximately reduce modulo pi/2 and then modulo pi/1024
	 double q = floor(r.x / _qd_pi2.x + 0.5);
	 gqd_real t = HP(r - mul_HD(_qd_pi2, q));
	 int j = (int) (q);
	 q = floor(t.x / _qd_pi1024.x + 0.5);
	 t = HP(t - mul_HD(_qd_pi1024, q));
	 int k = (int) (q);
	 int abs_k = abs(k);

	 if (j < -2 || j > 2) {
		  r.x = r.y = r.z = r.w = 0.0;
		  return r;
	 }

	 if (abs_k > 256) {
		  r.x = r.y = r.z = r.w = 0.0;
		  return r;
	 }

	 if (k == 0) {
		  switch (j) {
				case 0:
					 return sin_taylor(t);
				case 1:
					 return cos_taylor(t);
				case -1:
					 return negative(cos_taylor(t));
				default:
					 return negative(sin_taylor(t));
		  }
	 }

	 sincos_taylor(t, &z, &r);

	 if (j == 0) {
		  z = HP(d_cos_table_qd[abs_k - 1] * z);
		  r = HP(d_sin_table_qd[abs_k - 1] * r);
		  if (k > 0) return HP(z + r);
		  else return HP(z - r);
	 } else if (j == 1) {
		  r = HP(d_cos_table_qd[abs_k - 1] * r);
		  z = HP(d_sin_table_qd[abs_k - 1] * z);
		  if (k > 0) return HP(r - z);
		  else return HP(r + z);
	 } else if (j == -1) {
		  z = HP(d_sin_table_qd[abs_k - 1] * z);
		  r = HP(d_cos_table_qd[abs_k - 1] * r);
		  if (k > 0) {
				return HP(z - r);
		  } else {
				r.x = -r.x;
				r.y = -r.y;
				r.z = -r.z;
				r.w = -r.w;
				return HP(r - z);
		  }
	 } else {
		  r = HP(d_sin_table_qd[abs_k - 1] * r);
		  z = HP(d_cos_table_qd[abs_k - 1] * z);
		  if (k > 0) {
				z.x = -z.x;
				z.y = -z.y;
				z.z = -z.z;
				z.w = -z.w;
				//r = d_sin_table[abs_k-1] * r;
				return HP(z - r);
		  } else {
				return HP(r - z);
		  }
	 }
}

__attribute__((overloadable))
gqd_real cos(const gqd_real a) {
#ifdef CosAsSin
	return sin(add_HH(a, _qd_pi2));
#else
	//not use(we are NOT on CPU) if (is_zero(a)) return One;

	 // approximately reduce modulo 2*pi
	 gqd_real z = nint(HP(a / _qd_2pi));
	 gqd_real r = HP(a - _qd_2pi * z);

	 // approximately reduce modulo pi/2 and then modulo pi/1024
	 double q = floor(r.x / _qd_pi2.x + 0.5);
	 gqd_real t = HP(r - mul_HD(_qd_pi2, q));
	 int j = (int) (q);
	 q = floor(t.x / _qd_pi1024.x + 0.5);
	 t = HP(t - mul_HD(_qd_pi1024, q));
	 int k = (int) (q);
	 int abs_k = abs(k);

	 if (j < -2 || j > 2) {
		  return Zero;
	 }

	 if (abs_k > 256) {
		  return Zero;
	 }

	 if (k == 0) {
		  switch (j) {
				case 0:
					 return cos_taylor(t);
				case 1:
					 return negative(sin_taylor(t));
				case -1:
					 return sin_taylor(t);
				default:
					 return negative(cos_taylor(t));
		  }
	 }

	 gqd_real sin_t, cos_t;
	 sincos_taylor(t, &sin_t, &cos_t);

	 gqd_real u = d_cos_table_qd[abs_k - 1];
	 gqd_real v = d_sin_table_qd[abs_k - 1];

	 if (j == 0) {
		  if (k > 0) {
				r = HP(u * cos_t - v * sin_t);
		  } else {
				r = HP(u * cos_t + v * sin_t);
		  }
	 } else if (j == 1) {
		  if (k > 0) {
				r = HP(negative(u) * sin_t - v * cos_t);
		  } else {
				r = HP(v * cos_t - u * sin_t);
		  }
	 } else if (j == -1) {
		  if (k > 0) {
				r = HP(u * sin_t + v * cos_t);
		  } else {
				r = HP(u * sin_t - v * cos_t);
		  }
	 } else {
		  if (k > 0) {
				r = HP(v * sin_t - u * cos_t);
		  } else {
				r = HP(negative(u) * cos_t - v * sin_t);
		  }
	 }

	 return r;
#endif
}
#ifdef UseTan
void sincos(const gqd_real a, gqd_real *sin_a, gqd_real *cos_a)
{
	//not use(we are NOT on CPU)
	/*if (is_zero(a))
	{
		*sin_a = Zero;
		*cos_a = One;
		return;
	}*/

	 // approximately reduce by 2*pi
	 gqd_real z = nint(HP(a / _qd_2pi));
	 gqd_real t = HP(a - _qd_2pi * z);

	 // approximately reduce by pi/2 and then by pi/1024.
	 double q = floor(t.x / _qd_pi2.x + 0.5);
	 t = HP(t - mul_HD(_qd_pi2, q));
	 int j = (int) (q);
	 q = floor(t.x / _qd_pi1024.x + 0.5);
	 t = HP(t - mul_HD(_qd_pi1024, q));
	 int k = (int) (q);
	 int abs_k = abs(k);

	 if (j < -2 || j > 2) {
		  *cos_a = *sin_a = Zero;
		  return;
	 }

	 if (abs_k > 256) {
		  *cos_a = *sin_a = Zero;
		  return;
	 }

	 gqd_real sin_t, cos_t;
	 sincos_taylor(t, &sin_t, &cos_t);

	 if (k == 0) {
		  if (j == 0) {
				*sin_a = sin_t;
				*cos_a = cos_t;
		  } else if (j == 1) {
				*sin_a = cos_t;
				*cos_a = negative(sin_t);
		  } else if (j == -1) {
				*sin_a = negative(cos_t);
				*cos_a = sin_t;
		  } else {
				*sin_a = negative(sin_t);
				*cos_a = negative(cos_t);
		  }
		  return;
	 }

	 gqd_real u = d_cos_table_qd[abs_k - 1];
	 gqd_real v = d_sin_table_qd[abs_k - 1];

	 if (j == 0) {
		  if (k > 0) {
				*sin_a = HP(u * sin_t + v * cos_t);
				*cos_a = HP(u * cos_t - v * sin_t);
		  } else {
				*sin_a = HP(u * sin_t - v * cos_t);
				*cos_a = HP(u * cos_t + v * sin_t);
		  }
	 } else if (j == 1) {
		  if (k > 0) {
				*cos_a = HP(negative(u) * sin_t - v * cos_t);
				*sin_a = HP(u * cos_t - v * sin_t);
		  } else {
				*cos_a = HP(v * cos_t - u * sin_t);
				*sin_a = HP(u * cos_t + v * sin_t);
		  }
	 } else if (j == -1) {
		  if (k > 0) {
				*cos_a = HP(u * sin_t + v * cos_t);
				*sin_a = HP(v * sin_t - u * cos_t);
		  } else {
				*cos_a = HP(u * sin_t - v * cos_t);
				*sin_a = HP(negative(u) * cos_t - v * sin_t);
		  }
	 } else {
		  if (k > 0) {
				*sin_a = HP(negative(u) * sin_t - v * cos_t);
				*cos_a = HP(v * sin_t - u * cos_t);
		  } else {
				*sin_a = HP(v * cos_t - u * sin_t);
				*cos_a = HP(negative(u) * cos_t - v * sin_t);
		  }
	 }
}

__attribute__((overloadable))
gqd_real tan(const gqd_real a)
{
	 gqd_real s, c;
	 sincos(a, &s, &c);
	 return HP(s / c);
}
#endif
";
	}
}

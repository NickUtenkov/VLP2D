using DD128Numeric;
using QD256Numeric;
using System;
using System.Numerics;

namespace VLP2D.Common
{
	internal class AlfaСonvergentUpperBound<T> where T : INumberBase<T>
	{
		int _2tm1 = 0;

		public AlfaСonvergentUpperBound()
		{
			_2tm1 = get_2tm1();
		}

		public static int get_2tm1()
		{
			int t = 0;
			if (typeof(T) == typeof(float)) t = 7;
			if (typeof(T) == typeof(double)) t = 14;
			if (typeof(T) == typeof(DD128)) t = 31;
			if (typeof(T) == typeof(QD256)) t = 62;
			return 2 * (t - 1);
		}

		public int upperBound(T diagElemIn, int upper)
		{
			//Malcolm Palmer "A fast method for solving a class of tridiagonal linear systems" (5)
			//(1/u^2)^i > β^(1 - t)
			//i * logᵦ(1/u^2) < 1 - t
			//i * (-2 *logᵦ(u)) < 1 - t
			//t - 1 < 2 *logᵦ(u) * i
			//i > (t - 1) / (2 * logᵦ(u))
			//i > (t - 1) / (2 * log₂(u) / log₂(β))
			//i > (t - 1) / (2 * log₂(u) / log₂(β))
			//i > (t - 1) / (2 * log₂(u) / 4)
			//i > (t - 1) / log₂(u) / 2
			//i > (t - 1) / (log₂(u) / 2)
			//i > 2*(t - 1) / log₂(u)
			float diagElem = float.CreateTruncating(diagElemIn);
			float sign = MathF.Sign(diagElem);
			float u = (diagElem + sign * MathF.Sqrt(diagElem * diagElem - 4.0f)) / 2.0f;
			float K = _2tm1 / MathF.Log2(u);//T.Log(u, β) == T.Log2(u) / 4; β == 16
			return Math.Min(upper, (int)K);
		}

		public static U alfaСonvergent<U>(U diagElem) where U : INumberBase<U>, IRootFunctions<U>
		{
			return diagElem / U.CreateTruncating(2) - U.Sqrt(diagElem * diagElem / U.CreateTruncating(4) - U.One);//[IL] p.54, top
		}
	}
}

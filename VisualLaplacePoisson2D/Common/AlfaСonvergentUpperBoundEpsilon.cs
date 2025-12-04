using System;
using System.Numerics;

namespace VLP2D.Common
{
	internal class AlfaСonvergentUpperBoundEpsilon
	{
		double l1;

		public AlfaСonvergentUpperBoundEpsilon(double eps)
		{
			l1 = Math.Log10(eps);
		}

		public int upperBound<T>(T diagElem, int upper) where T : INumberBase<T>
		{
			return upperBound(double.CreateTruncating(diagElem), upper);
		}

		public int upperBound(double diagElem, int upper)
		{//Solving the Poisson equation on the FPS-164.pdf, p12(31)
		 //for diag=2.000000001 alfa convergents at 178_502, eps=5E-14
		 //fir diag=2.00000000001 alfa convergents at 1_057_274, eps=5E-14
			if (diagElem == 2.0) return upper;//|diagElem| < 2 + 0.1E-16(or E-15)
			double D = Math.Sqrt(diagElem * diagElem - 4.0);
			double x1 = (diagElem + D) / 2.0;
			double x2 = (diagElem - D) / 2.0;
			double ω = Math.Abs(x1) > Math.Abs(x2) ? x2 / x1 : x1 / x2;
			double l2 = Math.Log10(ω);
			return Math.Min(upper, (int)(l1 / l2) + 1);
		}
	}
}

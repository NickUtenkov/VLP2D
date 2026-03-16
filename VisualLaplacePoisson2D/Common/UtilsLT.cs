using System;
using System.Collections.Generic;
using System.Numerics;

namespace VLP2D.Common
{
	//https://bitbucket.org/alexander_belov/sufarec/downloads/
	//Программы SuFaReC для сверхбыстрого расчета эллиптических уравнений в прямоугольной области.pdf
	//Белов Калиткин Эволюционная факторизация и сверхбыстрый счет на установление.pdf
	internal class UtilsLT<T> where T : INumber<T>, IPowerFunctions<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>
	{
		static T _05 = T.CreateTruncating(0.5), _2 = T.CreateTruncating(2), _4 = T.CreateTruncating(4);

		static public T[] calculateTau(T eps, T hx2, T hy2, T lengthX, T lengthY)
		{
			T[] lambda_max, lambda_min;
			T conditioning;
			(lambda_max, lambda_min, conditioning) = spectrumEstimates(hx2, hy2, lengthX, lengthY);
			T epsilon_background = T.CreateTruncating(UtilsEps.epsilon<T>()) * conditioning;//can use UtilsEps.epsilonBackground<T>()
			T eps_grid_sys = eps;
			T epsilon = T.Max(epsilon_background, eps_grid_sys);
			int[] sAll = numberOfSteps(double.CreateTruncating(conditioning), double.CreateTruncating(epsilon));
			List<T> listτ = new List<T>();
			for (int i = 0; i < sAll.Length; i++)
			{
				int S = sAll[i];
				T[] tau_lt = logarithmicSet(lambda_max, lambda_min, S);
				if (i == 0) listτ.AddRange(tau_lt);
				else for (int s = 1; s <= S / 2; s++) listτ.Add(tau_lt[2 * s - 1]);
			}

			return listτ.ToArray();

			static (T[], T[], T) spectrumEstimates(T hx2, T hy2, T lengthX, T lengthY)
			{//Constructs estimates for spectrum boundaries of the grid system
				T est_x = _4 / hx2;
				T est_y = _4 / hy2;
				T[] lambda_max = [est_x, est_y];
				T pilx = T.Pi / lengthX;
				T pily = T.Pi / lengthY;
				T[] lambda_min = [pilx * pilx, pily * pily];
				T conditioning = (lambda_max[0] + lambda_max[1]) / (lambda_min[0] + lambda_min[1]);
				return (lambda_max, lambda_min, conditioning);
			}

			static int[] numberOfSteps(double relation, double epsilon)
			{// Calculates the numbers of steps in nested logarithmic grids via a priori convergence estimate
				int count = 10;
				double[] S_temp = new double[count];
				int I = 0;
				S_temp[0] = double.Ceiling(-(4 / (Math.PI * Math.PI + 2 * Math.PI)) * Math.Log(relation) * Math.Log(epsilon));
				while (S_temp[I] >= 2)
				{
					S_temp[I + 1] = S_temp[I] / 2;
					I = I + 1;
					if (I == count) break;
				}

				int[] sAll = new int[I];
				double ceil = double.Ceiling(S_temp[I]);
				for (int m = 0; m < I; m++) sAll[m] = (int)(ceil * Math.Pow(2, m));

				return sAll;
			}

			static T[] logarithmicSet(T[] lambda_max, T[] lambda_min, int S)
			{// Constructs linear-trigonometric set in logarithmic scale with given number of steps
				T τMin = _2 / (lambda_max[0] + lambda_max[1]);
				T τMax = _2 / (lambda_min[0] + lambda_min[1]);
				T center = _05 * T.Log(τMin * τMax);
				T width = _05 * T.Log(τMax / τMin);
				T piPlus2 = T.Pi + _2;
				T _2Div = _2 / piPlus2;
				T piDiv = T.Pi / piPlus2;

				T[] τ = new T[S + 1];
				T _S = T.CreateTruncating(S);
				for (int s = 0; s <= S; s++)
				{
					T θ = T.CreateTruncating(s) / _S;
					T lt = _2Div * T.Cos(θ * T.Pi - T.Pi) + piDiv * (_2 * θ - T.One);
					τ[s] = T.Pow(T.E, center + width * lt);
				}

				return τ;
			}
		}
	}
}

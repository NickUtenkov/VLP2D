using System;
using System.Numerics;
using System.Threading.Tasks;
using VLP2D.Common;

namespace VLP2D.Model
{
	class SplittingScheme<T> : ProgonkaScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>
	{
		public SplittingScheme(int cXSegments, int cYSegments, T stepX, T stepY, T eps, Func<T, T, T> fKsi, ParallelOptions optionsParallelIn) :
			base(cXSegments, cYSegments, stepX, stepY, eps, fKsi, optionsParallelIn)
		{//S_VVCM p.258 (27) σ₁,σ₂
			T σ1 = T.CreateTruncating(0.5);
			T σ2 = T.CreateTruncating(0.5);
			T diagExtraX = stepX2 / (dt * σ1);
			T diagExtraY = stepY2 / (dt * σ2);

			calcAlpha(_2 + diagExtraX, _2 + diagExtraY);

			if (fKsi != null)
			{
				funcX = (i, j) => stepX2 * fn[i, j] / (σ1 * _2);
				funcY = (i, j) => stepY2 * fn[i, j] / (σ2 * _2);
			}
			else funcX = funcY = (i, j) => T.Zero;

			rhsX = (src, i, j, iter) => src[i, j] * diagExtraX + (T.One - σ1) * operatorLxx(src, i, j) / σ1 + funcX(i, j);
			rhsY = (src, i, j, iter) => src[i, j] * diagExtraY + (T.One - σ2) * operatorLyy(src, i, j) / σ2 + funcY(i, j);
		}
	}
}

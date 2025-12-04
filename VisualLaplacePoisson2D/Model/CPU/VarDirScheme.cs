using System;
using System.Numerics;
using System.Threading.Tasks;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VarDirScheme<T> : ProgonkaScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>
	{
		JordanSpeedup<T> jrd;

		public VarDirScheme(int cXSegments, int cYSegments, T stepX, T stepY, T eps, Func<T, T, T> fKsi, ParallelOptions optionsParallelIn, bool isJordan) :
			base(cXSegments, cYSegments, stepX, stepY, eps, fKsi, optionsParallelIn)
		{//http://ikt.muctr.ru/html2/11/lek11_5.html
			T stepX2DivY2 = stepX2 / stepY2;
			T stepY2DivX2 = stepY2 / stepX2;

			if (fKsi != null)
			{
				funcX = (i, j) => stepX2 * fn[i, j];
				funcY = (i, j) => stepY2 * fn[i, j];
			}
			else funcX = funcY = (i, j) => T.Zero;

			if (!isJordan)
			{
				T ω1 = stepX2 * _2 / dt;
				T ω2 = stepY2 * _2 / dt;
				/*
				(T δ1, T Δ1) = JordanSpeedup<T>.operatorBoundaries(cXSegments, stepX2);//[SNR] p.441, at middle
				(T δ2, T Δ2) = JordanSpeedup<T>.operatorBoundaries(cYSegments, stepY2);
				T ω1 = T.One / T.Sqrt(δ1 * Δ1);
				T ω2 = T.One / T.Sqrt(δ2 * Δ2);*/

				calcAlpha(ω1 + _2, ω2 + _2);

				rhsX = (src, i, j, iter) => src[i, j] * ω1 + stepX2DivY2 * operatorLyy(src, i, j) + funcX(i, j);
				rhsY = (src, i, j, iter) => src[i, j] * ω2 + stepY2DivX2 * operatorLxx(src, i, j) + funcY(i, j);
			}
			else
			{
				jrd = new JordanSpeedup<T>(cXSegments, cYSegments, stepX2, stepY2, eps);

				rhsX = (src, i, j, iter) => src[i, j] * jrd.w1(iter) * stepX2 + stepX2DivY2 * operatorLyy(src, i, j) + funcX(i, j);
				rhsY = (src, i, j, iter) => src[i, j] * jrd.w2(iter) * stepY2 + stepY2DivX2 * operatorLxx(src, i, j) + funcY(i, j);

				calculateIterationAlpha = calcVariableDirectionsMethodAlpha;
				bProgonkaFixedIters = true;
			}
		}

		public override int maxIterations() { return (jrd != null) ? jrd.maxIters : 0; }

		public override void cleanup()
		{
			jrd = null;
			base.cleanup();
		}

		void calcVariableDirectionsMethodAlpha(int iter)
		{
			alphaX[0] = T.Zero;
			T w1kPlus2 = stepX2 * jrd.w1(iter) + _2;
			kX = αCC.upperBound(w1kPlus2, cXSegments - 1);
			for (int i = 1; i <= kX; i++) alphaX[i] = T.One / (w1kPlus2 - alphaX[i - 1]);//[SNR] p.443, top

			alphaY[0] = T.Zero;
			T w2kPlus2 = stepY2 * jrd.w2(iter) + _2;
			kY = αCC.upperBound(w2kPlus2, cYSegments - 1);
			for (int i = 1; i <= kY; i++) alphaY[i] = T.One / (w2kPlus2 - alphaY[i - 1]);
		}

		public override IterationsKind iterationsKind()
		{
			return (jrd != null) ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}
	}
}

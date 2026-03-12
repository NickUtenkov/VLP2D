using System;
using System.Numerics;

namespace VLP2D.Model
{
	class VarDirScheme<T> : ProgonkaScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>
	{
		JordanSpeedup<T> jrd;

		public VarDirScheme(RectangleData<T> rectData, T eps, Func<T, T, T> fKsi, bool isJordan) :
			base(rectData.cXSegments, rectData.cYSegments, rectData.xMin, rectData.yMin, rectData.stepX, rectData.stepY, eps, fKsi)
		{//http://ikt.muctr.ru/html2/11/lek11_5.html
			T OneDivY2 = T.One / stepY2;
			T OneDivX2 = T.One / stepX2;

			if (fKsi != null)
			{
				funcX = (i, j) => fn[i, j];
				funcY = (i, j) => fn[i, j];
			}
			else funcX = funcY = (i, j) => T.Zero;

			T ω = _2 / dt;

			if (!isJordan) calcAlpha(_2 + stepX2 * ω, _2 + stepY2 * ω);
			else
			{
				jrd = new JordanSpeedup<T>(cXSegments, cYSegments, stepX2, stepY2, eps);

				calculateIterationAlpha = calcVariableDirectionsMethodAlpha;
				bProgonkaFixedIters = true;
			}
			Func<int, T> multiplierX = (iter) => !isJordan ? ω : jrd.w1(iter);
			Func<int, T> multiplierY = (iter) => !isJordan ? ω : jrd.w2(iter);
			rhsX = (src, i, j, iter) => stepX2 * (src[i, j] * multiplierX(iter) + operatorLyy(src, i, j) * OneDivY2 + funcX(i, j));
			rhsY = (src, i, j, iter) => stepY2 * (src[i, j] * multiplierY(iter) + operatorLxx(src, i, j) * OneDivX2 + funcY(i, j));
		}

		public override int maxIterations() { return (jrd != null) ? jrd.maxIters : 0; }

		public override void cleanup()
		{
			jrd = null;
			base.cleanup();
		}

		void calcVariableDirectionsMethodAlpha(int iter) => calcAlpha(jrd.w1(iter) * stepX2 + _2, jrd.w2(iter) * stepY2 + _2);

		public override IterationsKind iterationsKind()
		{
			return (jrd != null) ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}
	}
}

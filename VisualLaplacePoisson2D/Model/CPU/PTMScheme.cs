using System;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class PTMScheme<T> : PTMSchemeBase<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		T[,] rk, wk;//==0 on boundary
		readonly Action<int,int> funcFk;
		T stepX, stepY;
		int upper1, upper2;
		T _2 = T.CreateTruncating(2);

		public PTMScheme(int cXSegments, int cYSegments, T stepX, T stepY, bool isChebyshIn, T epsIn, Func<T, T, T> fKsi) :
			base(cXSegments, cYSegments, stepX, stepY, isChebyshIn, epsIn, fKsi)
		{
			rk = new T[cXSegments + 1, cYSegments + 1];
			wk = new T[cXSegments + 1, cYSegments + 1];
			if (fKsi == null) funcFk = funcFkLap;
			else funcFk = funcFkPoi;
			this.stepX = stepX;
			this.stepY = stepY;
			upper1 = rk.GetUpperBound(0);
			upper2 = rk.GetUpperBound(1);
		}

		void funcFkLap(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2);
		void funcFkPoi(int i, int j) => rk[i, j] = -UtilsOpLap.operatorLaplaceXY(un0, i, j, stepX2, stepY2, _2) - fn[i, j];//fn is NOT multiplied by step2 
		void funcWuk(int i, int j) => wk[i, j] = ksiDivider * (ksi1 * wk[i - 1, j] + ksi2 * wk[i, j - 1] + rk[i, j]);//Pakulina direchlet_num.pdf,p.11(25)
		void funcWk(int i, int j) => wk[i, j] = ksiDivider * (ksi1 * wk[i + 1, j] + ksi2 * wk[i, j + 1] + wk[i, j]);//Pakulina direchlet_num.pdf,p.11(26)
		void funcV(int i, int j) => un1[i, j] = un0[i, j] - tau1 * wk[i, j];//Pakulina direchlet_num.pdf,p.11(27)

		override public T doIteration(int iter)//poperemenno-treugol'nii method
		{
			GridIterator.iterate(upper1, upper2, funcFk);

			GridIterator.iterateSequent(wk, funcWuk);
			GridIterator.iterateReverseSequent(wk, funcWk);

			tau1 = isChebysh ? tauk[iter] : tau;

			T rc = T.Zero;
			if (!isChebysh) rc = GridIterator.iterateForMaxWithEps(un1.GetUpperBound(0), un1.GetUpperBound(1), funcV, (i, j) => T.Abs(un0[i, j] - un1[i, j]), eps);
			else
			{
				GridIterator.iterate(upper1, upper2, funcV);//no need to calc delta - fixed number of iterations
				rc = eps + eps;
			}

			UtilsSwap.swap(ref un0, ref un1);

			return rc;
		}
	}
}

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace VLP2D.Model
{
	public class JordanSpeedup<T> where T : INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IPowerFunctions<T>
	{
		readonly T[] w1k, w2k;
		public readonly int maxIters;

		public JordanSpeedup(int cXSegments, int cYSegments, T stepX2, T stepY2, T ε)
		{
			T _1 = T.One;
			T _2 = T.CreateTruncating(2.0);
			T _4 = T.CreateTruncating(4.0);
			T π = T.Pi;

			(T δ1, T Δ1) = operatorBoundaries(cXSegments, stepX2);//[SNR] p.441, at middle
			(T δ2, T Δ2) = operatorBoundaries(cYSegments, stepY2);

			T a = T.Sqrt((Δ1 - δ1) * (Δ2 - δ2) / ((Δ1 + δ2) * (Δ2 + δ1)));//[SNR] p.437 (21)
			T η = (_1 - a) / (_1 + a);//[SNR] p.437 (21)

			maxIters = int.CreateTruncating(double.Ceiling(double.CreateTruncating(T.Log(_4 / η, T.E) * T.Log(_4 / ε, T.E) / (π * π))));//[SNR] p.439 (28)
			w1k = new T[maxIters];
			w2k = new T[maxIters];

			T b = a * (Δ2 + δ1) / (Δ1 - δ1);//[SNR] p.437 (22)
			T t = (_1 - b) / (_1 + b);//[SNR] p.437 (22)

			T r = (Δ2 + Δ1 * b) / (_1 + b);//[SNR] p.437 (23)
			T s = (Δ2 - Δ1 * b) / (_1 + b);//[SNR] p.437 (24)

			T q = η * η * (_1 + η * η / _2) / T.CreateTruncating(16);//[SNR] p.440,at top
			T sqη = T.Sqrt(η);
			T[] μᵢ = w1k;
			int halfIters = maxIters / 2;
			T _maxIters = T.CreateTruncating(maxIters);
			for (int i = halfIters + 1; i <= maxIters; i++)
			{
				T σᵢ = (_2 * T.CreateTruncating(i) - _1) / (_2 * _maxIters);
				T q1 = T.Pow(q, (_2 * σᵢ - _1) / _4);
				T q2 = _1 + T.Pow(q, _1 - σᵢ) + T.Pow(q, _1 + σᵢ);
				T q3 = _1 + T.Pow(q, σᵢ) + T.Pow(q, _2 - σᵢ);
				T q4 = q2 / q3;
				μᵢ[i - 1] = sqη * q1 * q4;//μᵢ,[SNR] p.440,at top 
			}
			for (int i = 1; i <= halfIters; i++) μᵢ[i - 1] = η / μᵢ[maxIters - i];//==ksiJ
			for (int i = 0; i < maxIters; i++) w2k[i] = w1k[i];//==ksiJ

			for (int i = 0; i < maxIters; i++) w1k[i] = (r * w1k[i] + s) / (_1 + t * w1k[i]);//[SNR] p.439, at top
			for (int i = 0; i < maxIters; i++) w2k[i] = (r * w2k[i] - s) / (_1 - t * w2k[i]);//[SNR] p.439, at top
			//for (int i = 0; i < maxIters; i++) w1k[i] = T.One / T.Sqrt(δ1 * Δ1);//[SNR] p.439, at top
			//for (int i = 0; i < maxIters; i++) w2k[i] = T.One / T.Sqrt(δ2 * Δ2);//[SNR] p.439, at top
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public T w1(int idx) => w1k[idx];//ω looks like w, so use w

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public T w2(int idx) => w2k[idx];

		public static Tuple<T, T> operatorBoundaries(int cSegments, T step2)
		{
			T _2 = T.CreateTruncating(2.0);
			T _4 = T.CreateTruncating(4.0);
			T π = T.Pi;

			T _cSegments = T.CreateTruncating(cSegments);

			T sin = T.Sin(π / (_2 * _cSegments));
			T valMin = _4 / step2 * sin * sin;//[SNR] p.441, at middle

			T cos = T.Cos(π / (_2 * _cSegments));
			T valMax = _4 / step2 * cos * cos;//[SNR] p.441, at middle

			return new Tuple<T, T>(valMin, valMax);
		}
	}
}

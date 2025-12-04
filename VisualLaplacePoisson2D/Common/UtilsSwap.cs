using System;
using System.Runtime.CompilerServices;

namespace VLP2D.Common
{
	internal class UtilsSwap
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void swap<T>(ref T p1, ref T p2)
		{
			T pTmp = p1;
			p1 = p2;
			p2 = pTmp;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void swap<T>(Span<T> ar, int idx1, int idx2)
		{
			T tmp = ar[idx1];
			ar[idx1] = ar[idx2];
			ar[idx2] = tmp;
		}
	}
}

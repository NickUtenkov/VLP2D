namespace VLP2D.Common
{
	internal class UtilsLParam
	{
		public static int[] getFACRLParamArray(int n)
		{
			int pow = Utils.calculatePowOf2(n);
			if (pow == 0) pow = 1;
			if (pow > 4) pow = 4;
			int[] ar = new int[pow];
			for (int i = 0; i < pow; i++)
			{
				ar[i] = i;
			}
			return ar;
		}

		public static int[] getMarchingLParamArray(int n)
		{
			int powMax = Utils.calculatePowOf2((n - 1) / 2);
			int powMin = 0;// powMax - 4;
			if (powMin < 0) powMin = 0;
			int[] ar = new int[powMax - powMin];
			int val = 1 << powMin;
			for (int i = powMin; i < powMax; i++)
			{
				ar[i - powMin] = val;
				val <<= 1;
			}
			return ar;
		}
	}
}

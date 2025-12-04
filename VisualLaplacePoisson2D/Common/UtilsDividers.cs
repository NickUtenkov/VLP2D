using System;
using System.Collections.Generic;
using System.Linq;

namespace VLP2D.Common
{
	internal class UtilsDividers
	{
		public static bool containsDividers(int numb, int[] powers)
		{//based on simpleDividers
			bool proceed = true;
			while (proceed)
			{
				int limit = (int)Math.Sqrt(numb);
				proceed = false;
				for (int i = 2; i <= limit; i++)
				{
					if (numb % i == 0)
					{
						if (!powers.Contains(i)) return false;
						numb /= i;
						proceed = true;
						break;
					}
				}
			}
			if (!powers.Contains(numb)) return false;
			return true;
		}

		public static List<int> simpleDividers(int numb)
		{
			List<int> del = new List<int>();
			bool proceed = true;
			while (proceed)
			{
				int limit = (int)Math.Sqrt(numb);
				proceed = false;
				for (int i = 2; i <= limit; i++)
				{
					if (numb % i == 0)
					{
						if (!del.Contains(i)) del.Add(i);
						numb /= i;
						proceed = true;
						break;
					}
				}
			}
			if (!del.Contains(numb)) del.Add(numb);
			return del;
		}

		public static Dictionary<int, int> simpleDividersMap(int numb)
		{
			Dictionary<int, int> del = new Dictionary<int, int>();
			bool proceed = true;
			while (proceed)
			{
				int limit = (int)Math.Sqrt(numb);
				proceed = false;
				for (int i = 2; i <= limit; i++)
				{
					if (numb % i == 0)
					{
						if (!del.ContainsKey(i)) del.Add(i, 1);
						else del[i]++;
						numb /= i;
						proceed = true;
						break;
					}
				}
			}
			if (!del.ContainsKey(numb)) del.Add(numb, 1);
			else del[numb]++;
			return del;
		}

		public static List<int> allDividers(int numb)
		{
			List<int> del = new List<int>();
			int limit = (int)Math.Sqrt(numb);
			for (int i = 1; i <= limit; i++)
			{
				if (numb % i == 0)
				{
					del.Add(i);
					if (i * i != numb) del.Add(numb / i);
				}
			}
			return del;
		}
	}
}

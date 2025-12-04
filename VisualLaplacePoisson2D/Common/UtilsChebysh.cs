using System.Collections.Generic;

namespace VLP2D.Common
{
	internal class UtilsChebysh
	{
		enum Operation : int
		{
			AddElem = 1,
			DoubleList = 2,
		}

		static void doubleChebyshElements(int[] lst, ref int count, int delta)
		{
			int m4 = 4 * count + delta;
			for (int i = count - 1; i >= 0; i--)
			{
				lst[2 * i + 0] = lst[i];
				lst[2 * i + 1] = m4 - lst[i];
			}
			count *= 2;
		}

		public static int[] chebyshParams(int val)//Samarskii
		{
			int[] lst = new int[val];
			lst[0] = 1;
			int curCount = 1;
			List<Operation> actions = new List<Operation>();
			do
			{
				int rem = val % 2;
				if (rem == 1) val--;
				else val /= 2;
				actions.Add((rem == 1) ? Operation.AddElem : Operation.DoubleList);
			}
			while (val > 1);
			actions.Reverse();

			for (int i = 0; i < actions.Count; i++)
			{
				if (actions[i] == Operation.AddElem) lst[curCount] = ++curCount;
				else doubleChebyshElements(lst, ref curCount, (i < actions.Count - 1 && actions[i + 1] == Operation.AddElem) ? 2 : 0);
			}

			return lst;
		}

		static int[] doubleReductionElements(int[] lst1, int m)
		{
			int sz1 = lst1.Length;
			int[] lst2 = new int[sz1 * 2];
			for (int i = 0; i < sz1 / 2; i++)
			{
				lst2[4 * i + 0] = lst1[2 * i + 0] + 0;
				lst2[4 * i + 1] = lst1[2 * i + 0] + m;
				lst2[4 * i + 2] = lst1[2 * i + 1] + m;
				lst2[4 * i + 3] = lst1[2 * i + 1] + 0;
			}
			return lst2;
		}

		public static int[] reductionParams(int power)//[SNR] p.143;[SNE] p.144
		{
			if (power == 0) return [1];
			int[] lst = new int[2];//1 << power
			lst[0] = 2;
			lst[1] = 1;
			int nDoubles = power - 1;
			for (int i = 0; i < nDoubles; i++)
			{
				lst = doubleReductionElements(lst, 1 << (i + 1));
			}

			return lst;
		}
	}
}

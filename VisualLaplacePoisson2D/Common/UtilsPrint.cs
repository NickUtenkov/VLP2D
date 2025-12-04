using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VLP2D.Common
{
	internal class UtilsPrint
	{
		public static void printArray<T>(T[,] ar, string header, string format, int pad)
		{
#if DEBUG
			if (header != null) Trace.WriteLine(header);
			for (int i = 0; i <= ar.GetUpperBound(0); i++) printArrayRowFormattedWithPadLeft(ar, i, format, pad);
#endif
		}

		public static void printArrayRowFormattedWithPadLeft<T>(T[,] ar, int row, string format, int pad)
		{
			string s1 = string.Format(CultureInfo.InvariantCulture, format, ar[row, 0]).PadLeft(pad, ' ');
			for (int i = 1; i <= ar.GetUpperBound(1); i++) s1 += " " + string.Format(CultureInfo.InvariantCulture, format, ar[row, i]).PadLeft(pad, ' ');
			Trace.WriteLine(s1);
		}

		public static void printArray1D<T>(T[] ar, string format, int cols, string header)
		{
#if DEBUG
			if (!string.IsNullOrEmpty(header)) Debug.WriteLine(header);
			int j = 0;
			string s1 = string.Format(CultureInfo.InvariantCulture, format, ar[j]);
			for (j = 1; j <= cols; j++) s1 += string.Format(CultureInfo.InvariantCulture, " " + format, ar[j]);
			Debug.WriteLine(s1);
#endif
		}

		public static void printArray1DAs2D<T>(T[] ar, int rows, int cols, string header)
		{
#if DEBUG
			if (!string.IsNullOrEmpty(header)) Debug.WriteLine(header);
			for (int i = 0; i < rows; i++)
			{
				int j = 0;
				string s1 = string.Format(CultureInfo.InvariantCulture, "{0,9:0.000}", ar[i * cols + j]);
				for (j = 1; j < cols; j++) s1 += string.Format(CultureInfo.InvariantCulture, " {0,9:0.000}", ar[i * cols + j]);
				Debug.WriteLine(s1);
			}
#endif
		}

		public static void printArray1DAs2D<T>(IList<T> ar, int rows, int cols, string header)
		{
#if DEBUG
			if (!string.IsNullOrEmpty(header)) Debug.WriteLine(header);
			for (int i = 0; i < rows; i++)
			{
				int j = 0;
				string s1 = string.Format(CultureInfo.InvariantCulture, "{0,9:0.000}", ar[i * cols + j]);
				for (j = 1; j < cols; j++) s1 += string.Format(CultureInfo.InvariantCulture, " {0,9:0.000}", ar[i * cols + j]);
				Debug.WriteLine(s1);
			}
#endif
		}

		public static void printJaggedArray<T>(T[][] ar, string format, string header)
		{
			if (!string.IsNullOrEmpty(header)) Debug.WriteLine(header);
			for (int i = 0; i <= ar.GetUpperBound(0); i++) if (ar[i] != null) printArray1D(ar[i], format, ar[i].GetUpperBound(0), null);
		}
	}
}

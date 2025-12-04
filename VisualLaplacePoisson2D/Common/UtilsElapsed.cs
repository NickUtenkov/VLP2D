using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace VLP2D.Common
{
	using NamedElapsed = Tuple<string, float>;

	internal class UtilsElapsed
	{
		public static Stopwatch stopWatchEL;
		static List<NamedElapsed> listElapsed;

		public static float getExecutedSeconds(Stopwatch stopWatch, Action action)
		{
			stopWatch.Restart();
			action();
			stopWatch.Stop();

			return (float)stopWatch.Elapsed.TotalSeconds;
		}

		public static float getExecutedSeconds(Action action)
		{
			stopWatchEL.Restart();
			action();
			stopWatchEL.Stop();

			return (float)stopWatchEL.Elapsed.TotalSeconds;
		}

		public static void initElapsedList()
		{
			stopWatchEL = new Stopwatch();
			listElapsed = new List<NamedElapsed>();
		}

		public static void listElapsedAdd(string name, float elapsed)
		{
			listElapsed.Add(new NamedElapsed(name, elapsed));
		}

		public static string timesElapsed()
		{
			float sum = listElapsed.Sum(x => x.Item2);
			listElapsed.Sort((x, y) => x.Item2.CompareTo(y.Item2));
			double perc = 100.0 / sum;
			string strAll = "";
			for (int i = 0; i < listElapsed.Count; i++) strAll += string.Format(CultureInfo.InvariantCulture, "{0} {1:0.0#}% {2:0.0#} sec.\n", listElapsed[i].Item1, perc * listElapsed[i].Item2, listElapsed[i].Item2);
			strAll += string.Format(CultureInfo.InvariantCulture, "{0} {1:0.0#} sec.", "All", sum);
			return strAll;
		}
	}
}

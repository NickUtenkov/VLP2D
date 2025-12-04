using System;
using System.Threading;
using System.Windows;
using System.Windows.Threading;

namespace VLP2D.Common
{
	internal class UtilsThread
	{
		public static void runOnUIThread(Action agileCallback)
		{
#pragma warning disable 4014
			Application.Current.Dispatcher.Invoke(DispatcherPriority.Normal, agileCallback);
#pragma warning restore 4014
		}

		public static void runOnUIThreadAsync(Action agileCallback)
		{
#pragma warning disable 4014
			Application.Current.Dispatcher.BeginInvoke(DispatcherPriority.Background, agileCallback);
#pragma warning restore 4014
		}

		public static void runOnNewThread(Action action)
		{
			new Thread(new ThreadStart(delegate
			{
				action();
			})).Start();
		}
	}
}

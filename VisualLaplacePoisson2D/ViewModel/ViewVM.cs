using System;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;

namespace VLP2D.ViewModel
{
	internal class ViewVM : ObservableObject
	{
		const int initialWidth = 260;
		public ICommand toggleWidthCommand
		{
			get { return new DelegateCommand(() => toggleWidth()); }
		}

		int _inputWidth = initialWidth;
		public int inputWidth
		{
			get { return _inputWidth; }
			set
			{
				if (_inputWidth == value) return;
				_inputWidth = value;
				RaisePropertyChangedEvent("inputWidth");
			}
		}

		void toggleWidth()
		{
			int divider = 20, count = initialWidth / divider;
			int milliSecs = 600;
			int delay = milliSecs / count;
			Action<int> action;
			if (inputWidth == 0) action = (i) => inputWidth = divider * i;
			else action = (i) => inputWidth = divider * (count - i);
			Task.Run(() =>
			{
				for (int i = 0; i <= count; i++)
				{
					Application.Current.Dispatcher.Invoke(delegate { action(i); });
					Task.Delay(delay).Wait();
				}
			});
		}
	}
}

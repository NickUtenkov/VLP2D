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
			inputWidth = (inputWidth == 0) ? initialWidth : 0;
		}
	}
}

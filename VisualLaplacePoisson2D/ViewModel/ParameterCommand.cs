
using System;
using System.Diagnostics;
using System.Windows.Input;

namespace VLP2D.ViewModel
{
	public class ParameterCommand : ICommand
	{
		private readonly Action<object> _action;

		public ParameterCommand(Action<object> action)
		{
			_action = action;
		}

		public void Execute(object parameter)
		{
			_action(parameter);
		}

		public bool CanExecute(object parameter)
		{
			return true;
		}

		#pragma warning disable 67
		public event EventHandler CanExecuteChanged { add { } remove { } }
		#pragma warning restore 67
	}
}

using ELW.Library.Math;
using System;
using System.Diagnostics;
using System.Globalization;
using System.Windows;
using System.Windows.Threading;
using VLP2D.Model;

namespace VLP2D
{
	public partial class App : Application
	{
		private void Application_DispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
		{
			string strMsg = string.Format("{0}\n\n{1}", e.Exception.Message, "Complete the application?");
			MessageBoxResult res = MessageBox.Show(strMsg, "Error", MessageBoxButton.YesNo, MessageBoxImage.Stop);
			if (res == MessageBoxResult.No) e.Handled = true;
			else
			{
				e.Handled = false;
				Shutdown();
			}
		}

		//[HandleProcessCorruptedStateExceptions]
		void CurrentDomainUnhandledException_EventHandler(object sender, UnhandledExceptionEventArgs ee)
		{
			Exception ex = ee.ExceptionObject as Exception;
		}

		//[HandleProcessCorruptedStateExceptions]
		void DispatcherUnhandledException_EventHandler(object sender, DispatcherUnhandledExceptionEventArgs e)
		{
			string errorMessage = string.Format("An application error occurred. If this error occurs again there seems to be a serious bug in the application, and you better close it.\n\nError:{0}\n\nDo you want to continue?\n(if you click Yes you will continue with your work, if you click No the application will close)", e.Exception.Message);
			//insert code to log exception here
			if (MessageBox.Show(errorMessage, "Application UnhandledException Error", MessageBoxButton.YesNo, MessageBoxImage.Error) == MessageBoxResult.No)
			{
				if (MessageBox.Show("WARNING: The application will close. Any changes will not be saved!\nDo you really want to close it?", "Close the application!", MessageBoxButton.YesNoCancel, MessageBoxImage.Warning) == MessageBoxResult.Yes)
				{
					Application.Current.Shutdown();
				}
			}
			e.Handled = true;
		}

		protected override void OnStartup(StartupEventArgs e)
		{
			// Catch all unhandled exceptions in all threads.
			AppDomain.CurrentDomain.UnhandledException += CurrentDomainUnhandledException_EventHandler;
			// Catch all WPF unhandled exceptions.
			Dispatcher.CurrentDispatcher.UnhandledException += DispatcherUnhandledException_EventHandler;

			//AppDomain.CurrentDomain.AssemblyLoad += MyAssemblyLoadEventHandler;

			// Catch all handled exceptions in managed code, before the runtime searches the Call Stack 
			//AppDomain.CurrentDomain.FirstChanceException += FirstChanceException;

			// Catch all unobserved task exceptions.
			//TaskScheduler.UnobservedTaskException += UnobservedTaskException;

			CultureInfo ci = new CultureInfo(CultureInfo.InstalledUICulture.Name);
			if (ci.NumberFormat.NumberDecimalSeparator != ".") ci.NumberFormat.NumberDecimalSeparator = ".";
			VLP2D.Properties.Resources.Culture = ci;//doesn't help
		}

		/*static void MyAssemblyLoadEventHandler(object sender, AssemblyLoadEventArgs args)
		{
		}*/
	}
}

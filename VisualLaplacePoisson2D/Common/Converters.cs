
using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;

namespace VLP2D
{
	public class StringToVisibilityConverter  : IValueConverter
	{
		virtual public Object Convert(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			String str = (String)value;
			return ((str != null) && (str.Length>0)) ? Visibility.Visible : Visibility.Collapsed;
		}

		virtual public Object ConvertBack(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			throw new NotImplementedException("StringToVisibilityConverter::ConvertBack not implemented");
		}
	}

	public class IntToStringConverter : IValueConverter
	{
		virtual public Object Convert(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			return value.ToString();
		}

		virtual public Object ConvertBack(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			int rc = -1;
			Int32.TryParse((String)value, out rc);
			return rc;
		}
	}

	public class BoolToVisibilityConverter : IValueConverter
	{
		virtual public Object Convert(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			bool bVal = (bool)value;
			return bVal ? Visibility.Visible : Visibility.Collapsed;
		}

		virtual public Object ConvertBack(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			throw new NotImplementedException("BoolToVisibilityConverter::ConvertBack not implemented");
		}
	}

	public class BoolToNoVisibilityConverter : IValueConverter
	{
		virtual public Object Convert(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			bool bVal = (bool)value;
			return bVal ? Visibility.Collapsed : Visibility.Visible;
		}

		virtual public Object ConvertBack(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			throw new NotImplementedException("BoolToNoVisibilityConverter::ConvertBack not implemented");
		}
	}

	public class TextToVisibilityConverter : IValueConverter
	{
		virtual public Object Convert(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			bool bVal = (value != null) ? ((string)value).Length > 0 : false;
			return bVal ? Visibility.Visible : Visibility.Collapsed;
		}

		virtual public Object ConvertBack(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			throw new NotImplementedException("BoolToVisibilityConverter::ConvertBack not implemented");
		}
	}

	public class BoolToColor1Converter : IValueConverter
	{
		static SolidColorBrush clrTextSelected = new SolidColorBrush(Color.FromArgb(255, 192, 192, 192));
		static SolidColorBrush clrTextUnselected = new SolidColorBrush(Color.FromArgb(255, 255, 255, 255));
		virtual public Object Convert(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			bool bVal = (bool)value;
			return bVal ? clrTextSelected : clrTextUnselected;
		}

		virtual public Object ConvertBack(Object value, Type targetType, Object parameter, CultureInfo language)
		{
			throw new NotImplementedException("BoolToColor1Converter::ConvertBack not implemented");
		}
	}

	public class MultiBoolToNoVisibilityConverter : IMultiValueConverter
	{
		public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
		{
			return (bool)values[0] || (bool)values[1] ? Visibility.Collapsed : Visibility.Visible;
		}

		public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
		{
			throw new NotImplementedException();
		}
	}

	public class EnumConverter : IValueConverter
	{
		public object Convert(object value, Type targetType, object parameter,CultureInfo culture)
		{
			int returnValue = 0;
			if (parameter is Type)
			{
				returnValue = (int)Enum.Parse((Type)parameter, value.ToString());
			}
			return returnValue;
		}

		public object ConvertBack(object value, Type targetType, object parameter,CultureInfo culture)
		{
			Enum enumValue = default(Enum);
			if (parameter is Type)
			{
				enumValue = (Enum)Enum.Parse((Type)parameter, value.ToString());
			}
			return enumValue;
		}
	}
}

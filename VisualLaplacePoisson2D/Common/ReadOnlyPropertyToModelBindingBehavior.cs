using Microsoft.Xaml.Behaviors;
using System.Windows;

namespace VLP2D.Common
{//https://stackoverflow.com/questions/1083224/pushing-read-only-gui-properties-back-into-viewmodel
	public class ReadOnlyPropertyToModelBindingBehavior : Behavior<UIElement>
	{
		public object ReadOnlyDependencyProperty
		{
			get { return (object)GetValue(ReadOnlyDependencyPropertyProperty); }
			set { SetValue(ReadOnlyDependencyPropertyProperty, value); }
		}

		public static readonly DependencyProperty ReadOnlyDependencyPropertyProperty =
			 DependencyProperty.Register("ReadOnlyDependencyProperty", typeof(object), typeof(ReadOnlyPropertyToModelBindingBehavior),
				  new PropertyMetadata(null, OnReadOnlyDependencyPropertyPropertyChanged));

		public object ModelProperty
		{
			get { return (object)GetValue(ModelPropertyProperty); }
			set { SetValue(ModelPropertyProperty, value); }
		}

		public static readonly DependencyProperty ModelPropertyProperty =
			 DependencyProperty.Register("ModelProperty", typeof(object), typeof(ReadOnlyPropertyToModelBindingBehavior), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault));

		private static void OnReadOnlyDependencyPropertyPropertyChanged(DependencyObject obj, DependencyPropertyChangedEventArgs e)
		{
			var b = obj as ReadOnlyPropertyToModelBindingBehavior;
			b.ModelProperty = e.NewValue;
		}
	}
}

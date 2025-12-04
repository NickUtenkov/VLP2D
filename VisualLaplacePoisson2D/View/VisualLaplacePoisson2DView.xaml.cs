using VLP2D.ViewModel;
using System.Windows.Controls;

namespace VLP2D.View
{
	public partial class VisualLaplacePoisson2DView : UserControl
	{
		public VisualLaplacePoisson2DView()
		{
			InitializeComponent();
			{
				IVLPRectangleInput iVM = (IVLPRectangleInput)VLPRectangleInputView.DataContext;
				IVLPRectangleOutput oVM = (IVLPRectangleOutput)VLPRectangleOutputView.DataContext;
				iVM.setOutput(oVM);
				oVM.setInput(iVM);

				oVM.changeModelPrecision(iVM.precisionIndex());
			}
		}
	}
}

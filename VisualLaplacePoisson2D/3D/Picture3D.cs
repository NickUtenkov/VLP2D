using System;
using System.Collections;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using WPFChart3D;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.ViewModel
{
	using USChart3D = UniformSurfaceChart3D;

	internal class Picture3D
	{
		WPFChart3D.Model3D model;
		TransformMatrix transformMatrix;
		ParallelOptions optionsParallel;

		public Picture3D()
		{
			optionsParallel = new ParallelOptions();
			optionsParallel.MaxDegreeOfParallelism = Environment.ProcessorCount;
		}

		public System.Windows.Media.Media3D.Model3D plot(Adapter2D<float> adapter, MinMaxF rangeX, MinMaxF rangeY)
		{
			Chart3D chart = new USChart3D();
			((USChart3D)chart).SetGrid(adapter.dim1, adapter.dim2, rangeX.min, rangeX.max, rangeY.min, rangeY.max, optionsParallel);

			Parallel.For(0, adapter.dim1, optionsParallel, i =>
			{//order the same as in UniformSurfaceChart3D
				for (int j = 0; j < adapter.dim2; j++) chart[j * adapter.dim1 + i].z = adapter.func(i, j);
			});

			chart.GetDataRange();

			double zMin = chart.ZMin();
			double zMax = chart.ZMax();
			Parallel.For(0, adapter.dim1, optionsParallel, k =>
			{
				int i = k;
				for (int j = 0; j < adapter.dim2; j++)
				{
					chart[i].color = TextureMapping.PaletteColor(palNoPurple, chart[i].z, zMin, zMax);
					i += adapter.dim1;
				}
			});

			ArrayList meshs = ((USChart3D)chart).GetMeshes(optionsParallel);
			model = new WPFChart3D.Model3D();
			Material backMaterial = new DiffuseMaterial(new SolidColorBrush(Colors.AntiqueWhite));
			model.UpdateModel(meshs, backMaterial);

			transformMatrix = new TransformMatrix();
			transformMatrix.initViewMatrix();
			transformMatrix.CalculateProjectionMatrix(chart.XMin(), chart.XMax(), chart.YMin(), chart.YMax(), zMin, zMax, 0.5);
			TransformChart();

			return model.Content;
		}

		public void mouseDown(Point pt) => transformMatrix.OnLBtnDown(pt);
		public void mouseUp() => transformMatrix.OnLBtnUp();

		public void mouseMove(Point pt, double actualWidth, double actualHeight)
		{
			transformMatrix.OnMouseMove(pt, actualWidth, actualHeight);
			TransformChart();
		}

		public void home()
		{
			transformMatrix.home();
			TransformChart();
		}

		public void zoomIn()
		{
			transformMatrix.zoomIn();
			TransformChart();
		}

		public void zoomOut()
		{
			transformMatrix.zoomOut();
			TransformChart();
		}

		public void reset()
		{
			model = null;
			transformMatrix = null;
		}

		private void TransformChart()//rotate, drag and zoom the 3d chart
		{
			if (model.Content == null) return;
			Transform3DGroup group1 = model.Content.Transform as Transform3DGroup;
			group1.Children.Clear();
			group1.Children.Add(new MatrixTransform3D(transformMatrix.m_totalMatrix));
		}
	}
}

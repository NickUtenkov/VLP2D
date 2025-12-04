using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using static VLP2D.Common.Utils;

namespace VLP2D.Common
{
	public class UtilsPict
	{
		public static int pictDim = 400;
		public class MinMaxF
		{
			public float min;
			public float max;

			public MinMaxF()
			{
				min = float.NaN;
				max = float.NaN;
			}

			public MinMaxF(float min, float max)
			{
				this.min = min;
				this.max = max;
			}

			public MinMaxF(double min, double max)
			{
				this.min = (float)min;
				this.max = (float)max;
			}
		}

		public static BitmapSource createHeatMap(BitmapPalette pal, MinMaxF minMax, Adapter2D<float> adapter, int excludedValues = 0)//remove excludedValues ?!
		{//RenderOptions.BitmapScalingMode="NearestNeighbor"
			BitmapSource rc = null;
			float maxPalEntryValue = pal.Colors.Count - 1 - excludedValues;
			var pixels = new byte[adapter.dim1 * adapter.dim2];
			float df = minMax.max - minMax.min;
			for (int i = 0; i < adapter.dim1; i++)
			{
				for (int j = 0; j < adapter.dim2; j++)
				{
					float idx = !(adapter.func(i, j) == float.NaN) ? 1 + (maxPalEntryValue - 1) * (adapter.func(i, j) - minMax.min) / df + excludedValues : 0;
					pixels[i + (adapter.dim2 - 1 - j) * adapter.dim1] = (byte)idx;
				}
			}
			var stride = adapter.dim1;
			UtilsThread.runOnUIThread(() =>
			{
				rc = BitmapSource.Create(adapter.dim1, adapter.dim2, 96, 96, PixelFormats.Indexed8, pal, pixels, stride);
			});
			return rc;
		}

		public static BitmapSource createInterpolatedHeatMap(BitmapPalette pal, MinMaxF minMax, Adapter2D<float> adapter, float stepX, float stepY, int width, int height)
		{//https://ru.wikipedia.org/wiki/%D0%91%D0%B8%D0%BB%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F_%D0%B8%D0%BD%D1%82%D0%B5%D1%80%D0%BF%D0%BE%D0%BB%D1%8F%D1%86%D0%B8%D1%8F
		 //https://en.wikipedia.org/wiki/Bilinear_interpolation
			if (width < pictDim && height < pictDim)
			{
				return createHeatMap(pal, minMax, adapter, 0);
			}
			float fMin = float.MaxValue, fMax = float.MinValue;
			Action<float> updateMinMax = (val1) =>
			{
				if (!float.IsNaN(val1))
				{
					if (fMax < val1) fMax = val1;
					if (fMin > val1) fMin = val1;
				}
			};

			float lngXFine = (adapter.dim1 - 1) * stepX;
			float lngYFine = (adapter.dim2 - 1) * stepY;
			float[,] inter = new float[width, height];

			float hXFine = stepX;
			float hYFine = stepY;
			float e = 1 / (hXFine * hYFine);

			float lngXCoarse = lngXFine;
			float lngYCoarse = lngYFine;

			float hXCoarse = lngXCoarse / width;
			float hYCoarse = lngYCoarse / height;

			int[] xIndexesLeft = { 1, };
			int[] xIndexesRight = { width - 2, };
			Func<int, int, float> val = (i, j) =>
			{
				float val0 = adapter.func(i, j);
				return !float.IsNaN(val0) ? val0 : 0;
			};

			Parallel.For(0, width, i =>
			{
				float xx = hXCoarse * i;
				float xFine = xx / hXFine;
				int i1 = (int)Math.Truncate(xFine);
				int i2 = i1 + 1;
				float x1 = i1 * hXFine;
				float x2 = x1 + hXFine;

				for (int j = 0; j < height; j++)
				{
					float yy = hYCoarse * j;
					float yFine = yy / hYFine;
					int j1 = (int)Math.Truncate(yFine);
					int j2 = j1 + 1;
					float y1 = j1 * hYFine;
					float y2 = y1 + hYFine;

					float dx1 = xx - x1;
					float dx2 = x2 - xx;
					float dy1 = yy - y1;
					float dy2 = y2 - yy;
					if (Array.IndexOf(xIndexesLeft, i) != -1)
					{
						inter[i, j] = val(i1, j1) + (val(i1, j2) - val(i1, j1)) * (yy - y1) / stepY;
					}
					else if (Array.IndexOf(xIndexesRight, i) != -1)
					{
						inter[i, j] = val(i2, j1) + (val(i2, j2) - val(i2, j1)) * (yy - y1) / stepY;
					}
					else inter[i, j] = (val(i1, j1) * dx2 * dy2 + val(i2, j1) * dx1 * dy2 + val(i1, j2) * dx2 * dy1 + val(i2, j2) * dx1 * dy1) * e;
					updateMinMax(inter[i, j]);
				}
			}
			);

			return createHeatMap(pal, new MinMaxF(fMin, fMax), new Adapter2D<float>(width, height, (i, j) => inter[i, j]), 0);
		}

		public static void addPicture(List<BitmapSource> lstBitmap, bool palWithTransparent, MinMaxF minMax, Adapter2D<float> adapter, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			if (lstBitmap != null)
			{
				if (minMax == null)
				{
					double fMin = Double.MaxValue, fMax = Double.MinValue;
					Action<double> updateMinMax = (val) =>
					{
						if (fMax < val) fMax = val;
						if (fMin > val) fMin = val;
					};
					GridIterator.iterateWithEdges(adapter.dim1, adapter.dim2, (i, j) => updateMinMax(adapter.func(i, j)));
					minMax = new MinMaxF(fMin, fMax);
				}
				BitmapSource srcBmp = fCreateBitmap(palWithTransparent, minMax, adapter);
				lstBitmap.Add(srcBmp);
			}
		}
	}
}

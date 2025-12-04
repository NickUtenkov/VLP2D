using System;
using System.IO;
using System.Windows.Media.Imaging;

namespace VLP2D.Common
{
	public class FilmStrip
	{//https://torlanglo.wordpress.com/2006/12/16/creating-animated-gif-files-in-wpf/, ezgif.com - speed change
		private BitmapEncoder fEncoder;

		protected BitmapEncoder Encoder
		{
			get { return fEncoder; }
		}

		public void AddFrame(BitmapFrame frame)
		{
			Encoder.Frames.Add(frame);
		}

		public void AddFrame(BitmapSource source)
		{
			AddFrame(BitmapFrame.Create(source));
		}

		protected virtual BitmapEncoder CreateEncoder()
		{
			return new GifBitmapEncoder();
		}

		public void SaveToFile(String fileName)
		{
			using (FileStream fs = new FileStream(fileName, FileMode.Create))
			{
				Encoder.Save(fs);
			}
		}

		public void Start()
		{
			fEncoder = CreateEncoder();
		}
	}
}

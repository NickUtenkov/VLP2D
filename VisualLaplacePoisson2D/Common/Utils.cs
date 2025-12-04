using ELW.Library.Math;
using ELW.Library.Math.Calculators;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq.Expressions;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace VLP2D.Common
{
	public static class Utils
	{
		public static readonly BitmapPalette palNoPurple, palNoPurpleWithTransparent;

		public struct Adapter2D<T>
		{
			public int dim1, dim2;
			public Func<int, int, T> func;
			public Adapter2D(int dim1, int dim2, Func<int, int, T> func)
			{
				this.dim1 = dim1;
				this.dim2 = dim2;
				this.func = func;
			}
		}

		static Utils()
		{
			List<Color> palColors = new List<Color>
			{
				Color.FromRgb(0, 0, 255),
				Color.FromRgb(0, 4, 255),
				Color.FromRgb(0, 8, 255),
				Color.FromRgb(0, 12, 255),
				Color.FromRgb(0, 16, 255),
				Color.FromRgb(0, 20, 255),
				Color.FromRgb(0, 24, 255),
				Color.FromRgb(0, 28, 255),
				Color.FromRgb(0, 32, 255),
				Color.FromRgb(0, 36, 255),
				Color.FromRgb(0, 40, 255),
				Color.FromRgb(0, 44, 255),
				Color.FromRgb(0, 48, 255),
				Color.FromRgb(0, 52, 255),
				Color.FromRgb(0, 56, 255),
				Color.FromRgb(0, 60, 255),
				Color.FromRgb(0, 64, 255),
				Color.FromRgb(0, 68, 255),
				Color.FromRgb(0, 72, 255),
				Color.FromRgb(0, 76, 255),
				Color.FromRgb(0, 80, 255),
				Color.FromRgb(0, 84, 255),
				Color.FromRgb(0, 88, 255),
				Color.FromRgb(0, 92, 255),
				Color.FromRgb(0, 96, 255),
				Color.FromRgb(0, 100, 255),
				Color.FromRgb(0, 104, 255),
				Color.FromRgb(0, 108, 255),
				Color.FromRgb(0, 112, 255),
				Color.FromRgb(0, 116, 255),
				Color.FromRgb(0, 120, 255),
				Color.FromRgb(0, 124, 255),
				Color.FromRgb(0, 128, 255),
				Color.FromRgb(0, 132, 255),
				Color.FromRgb(0, 136, 255),
				Color.FromRgb(0, 140, 255),
				Color.FromRgb(0, 144, 255),
				Color.FromRgb(0, 148, 255),
				Color.FromRgb(0, 152, 255),
				Color.FromRgb(0, 156, 255),
				Color.FromRgb(0, 160, 255),
				Color.FromRgb(0, 164, 255),
				Color.FromRgb(0, 168, 255),
				Color.FromRgb(0, 172, 255),
				Color.FromRgb(0, 176, 255),
				Color.FromRgb(0, 180, 255),
				Color.FromRgb(0, 184, 255),
				Color.FromRgb(0, 188, 255),
				Color.FromRgb(0, 192, 255),
				Color.FromRgb(0, 196, 255),
				Color.FromRgb(0, 200, 255),
				Color.FromRgb(0, 204, 255),
				Color.FromRgb(0, 208, 255),
				Color.FromRgb(0, 212, 255),
				Color.FromRgb(0, 216, 255),
				Color.FromRgb(0, 220, 255),
				Color.FromRgb(0, 224, 255),
				Color.FromRgb(0, 228, 255),
				Color.FromRgb(0, 232, 255),
				Color.FromRgb(0, 236, 255),
				Color.FromRgb(0, 240, 255),
				Color.FromRgb(0, 244, 255),
				Color.FromRgb(0, 248, 255),
				Color.FromRgb(0, 252, 255),
				Color.FromRgb(0, 254, 253),
				Color.FromRgb(0, 252, 249),
				Color.FromRgb(0, 250, 245),
				Color.FromRgb(0, 248, 241),
				Color.FromRgb(0, 246, 237),
				Color.FromRgb(0, 244, 233),
				Color.FromRgb(0, 242, 229),
				Color.FromRgb(0, 240, 225),
				Color.FromRgb(0, 238, 221),
				Color.FromRgb(0, 236, 217),
				Color.FromRgb(0, 234, 213),
				Color.FromRgb(0, 232, 209),
				Color.FromRgb(0, 230, 205),
				Color.FromRgb(0, 228, 201),
				Color.FromRgb(0, 226, 197),
				Color.FromRgb(0, 224, 193),
				Color.FromRgb(0, 222, 189),
				Color.FromRgb(0, 220, 185),
				Color.FromRgb(0, 218, 181),
				Color.FromRgb(0, 216, 177),
				Color.FromRgb(0, 214, 173),
				Color.FromRgb(0, 212, 169),
				Color.FromRgb(0, 210, 165),
				Color.FromRgb(0, 208, 161),
				Color.FromRgb(0, 206, 157),
				Color.FromRgb(0, 204, 153),
				Color.FromRgb(0, 202, 149),
				Color.FromRgb(0, 200, 145),
				Color.FromRgb(0, 198, 141),
				Color.FromRgb(0, 196, 137),
				Color.FromRgb(0, 194, 133),
				Color.FromRgb(0, 192, 129),
				Color.FromRgb(0, 190, 125),
				Color.FromRgb(0, 188, 121),
				Color.FromRgb(0, 186, 117),
				Color.FromRgb(0, 184, 113),
				Color.FromRgb(0, 182, 109),
				Color.FromRgb(0, 180, 105),
				Color.FromRgb(0, 178, 101),
				Color.FromRgb(0, 176, 97),
				Color.FromRgb(0, 174, 93),
				Color.FromRgb(0, 172, 89),
				Color.FromRgb(0, 170, 85),
				Color.FromRgb(0, 168, 81),
				Color.FromRgb(0, 166, 77),
				Color.FromRgb(0, 164, 73),
				Color.FromRgb(0, 162, 69),
				Color.FromRgb(0, 160, 65),
				Color.FromRgb(0, 158, 61),
				Color.FromRgb(0, 156, 57),
				Color.FromRgb(0, 154, 53),
				Color.FromRgb(0, 152, 49),
				Color.FromRgb(0, 150, 45),
				Color.FromRgb(0, 148, 41),
				Color.FromRgb(0, 146, 37),
				Color.FromRgb(0, 144, 33),
				Color.FromRgb(0, 142, 29),
				Color.FromRgb(0, 140, 25),
				Color.FromRgb(0, 138, 21),
				Color.FromRgb(0, 136, 17),
				Color.FromRgb(0, 134, 13),
				Color.FromRgb(0, 132, 9),
				Color.FromRgb(0, 130, 5),
				Color.FromRgb(0, 128, 1),
				Color.FromRgb(2, 129, 0),
				Color.FromRgb(6, 131, 0),
				Color.FromRgb(10, 133, 0),
				Color.FromRgb(14, 135, 0),
				Color.FromRgb(18, 137, 0),
				Color.FromRgb(22, 139, 0),
				Color.FromRgb(26, 141, 0),
				Color.FromRgb(30, 143, 0),
				Color.FromRgb(34, 145, 0),
				Color.FromRgb(38, 147, 0),
				Color.FromRgb(42, 149, 0),
				Color.FromRgb(46, 151, 0),
				Color.FromRgb(50, 153, 0),
				Color.FromRgb(54, 155, 0),
				Color.FromRgb(58, 157, 0),
				Color.FromRgb(62, 159, 0),
				Color.FromRgb(66, 161, 0),
				Color.FromRgb(70, 163, 0),
				Color.FromRgb(74, 165, 0),
				Color.FromRgb(78, 167, 0),
				Color.FromRgb(82, 169, 0),
				Color.FromRgb(86, 170, 0),
				Color.FromRgb(90, 172, 0),
				Color.FromRgb(94, 174, 0),
				Color.FromRgb(98, 176, 0),
				Color.FromRgb(102, 178, 0),
				Color.FromRgb(106, 180, 0),
				Color.FromRgb(110, 182, 0),
				Color.FromRgb(114, 184, 0),
				Color.FromRgb(118, 186, 0),
				Color.FromRgb(122, 188, 0),
				Color.FromRgb(126, 190, 0),
				Color.FromRgb(130, 192, 0),
				Color.FromRgb(134, 194, 0),
				Color.FromRgb(138, 196, 0),
				Color.FromRgb(142, 198, 0),
				Color.FromRgb(146, 200, 0),
				Color.FromRgb(150, 202, 0),
				Color.FromRgb(154, 204, 0),
				Color.FromRgb(158, 206, 0),
				Color.FromRgb(162, 208, 0),
				Color.FromRgb(166, 210, 0),
				Color.FromRgb(170, 212, 0),
				Color.FromRgb(174, 214, 0),
				Color.FromRgb(178, 216, 0),
				Color.FromRgb(182, 218, 0),
				Color.FromRgb(186, 220, 0),
				Color.FromRgb(190, 222, 0),
				Color.FromRgb(194, 224, 0),
				Color.FromRgb(198, 226, 0),
				Color.FromRgb(202, 228, 0),
				Color.FromRgb(206, 230, 0),
				Color.FromRgb(210, 232, 0),
				Color.FromRgb(214, 234, 0),
				Color.FromRgb(218, 236, 0),
				Color.FromRgb(222, 238, 0),
				Color.FromRgb(226, 240, 0),
				Color.FromRgb(230, 242, 0),
				Color.FromRgb(234, 244, 0),
				Color.FromRgb(238, 246, 0),
				Color.FromRgb(242, 248, 0),
				Color.FromRgb(246, 250, 0),
				Color.FromRgb(250, 252, 0),
				Color.FromRgb(254, 254, 0),
				Color.FromRgb(255, 251, 0),
				Color.FromRgb(255, 247, 0),
				Color.FromRgb(255, 243, 0),
				Color.FromRgb(255, 239, 0),
				Color.FromRgb(255, 235, 0),
				Color.FromRgb(255, 231, 0),
				Color.FromRgb(255, 227, 0),
				Color.FromRgb(255, 223, 0),
				Color.FromRgb(255, 219, 0),
				Color.FromRgb(255, 215, 0),
				Color.FromRgb(255, 211, 0),
				Color.FromRgb(255, 207, 0),
				Color.FromRgb(255, 203, 0),
				Color.FromRgb(255, 199, 0),
				Color.FromRgb(255, 195, 0),
				Color.FromRgb(255, 191, 0),
				Color.FromRgb(255, 187, 0),
				Color.FromRgb(255, 183, 0),
				Color.FromRgb(255, 179, 0),
				Color.FromRgb(255, 175, 0),
				Color.FromRgb(255, 171, 0),
				Color.FromRgb(255, 167, 0),
				Color.FromRgb(255, 163, 0),
				Color.FromRgb(255, 159, 0),
				Color.FromRgb(255, 155, 0),
				Color.FromRgb(255, 151, 0),
				Color.FromRgb(255, 147, 0),
				Color.FromRgb(255, 143, 0),
				Color.FromRgb(255, 139, 0),
				Color.FromRgb(255, 135, 0),
				Color.FromRgb(255, 131, 0),
				Color.FromRgb(255, 127, 0),
				Color.FromRgb(255, 123, 0),
				Color.FromRgb(255, 119, 0),
				Color.FromRgb(255, 115, 0),
				Color.FromRgb(255, 111, 0),
				Color.FromRgb(255, 107, 0),
				Color.FromRgb(255, 103, 0),
				Color.FromRgb(255, 99, 0),
				Color.FromRgb(255, 95, 0),
				Color.FromRgb(255, 91, 0),
				Color.FromRgb(255, 87, 0),
				Color.FromRgb(255, 83, 0),
				Color.FromRgb(255, 79, 0),
				Color.FromRgb(255, 75, 0),
				Color.FromRgb(255, 71, 0),
				Color.FromRgb(255, 67, 0),
				Color.FromRgb(255, 63, 0),
				Color.FromRgb(255, 59, 0),
				Color.FromRgb(255, 55, 0),
				Color.FromRgb(255, 51, 0),
				Color.FromRgb(255, 47, 0),
				Color.FromRgb(255, 43, 0),
				Color.FromRgb(255, 39, 0),
				Color.FromRgb(255, 35, 0),
				Color.FromRgb(255, 31, 0),
				Color.FromRgb(255, 27, 0),
				Color.FromRgb(255, 23, 0),
				Color.FromRgb(255, 19, 0),
				Color.FromRgb(255, 15, 0),
				Color.FromRgb(255, 11, 0),
				Color.FromRgb(255, 7, 0),
				Color.FromRgb(255, 3, 0),
				Color.FromRgb(255, 0, 0),
			};
			palNoPurple = new BitmapPalette(palColors);

			palColors[0] = Color.FromArgb(0, 255, 255, 255);
			palNoPurpleWithTransparent = new BitmapPalette(palColors);
		}

		public static void generateSqrtCoefs<T>(int l, Action<int, T> action) where T : INumberBase<T>, IRootFunctions<T>//, IAdditionOperators<T, double, T>, IMultiplyOperators<T, double, T>
		{
			//replace C with 'C²-2' for next l param
			//l=1 C = 2*(1+α) [SNR] p.106 (4)
			//l=2 C²-2 = (C-√2)*(C+√2)
			//l=3 (C²-2)²-2 = (C²-2-√2)*(C²-2+√2) = (C²-(2+√2))*(C²-(2-√2)) = (C-√(2+√2)) * (C+√(2+√2)) * (C-√(2-√2)) * (C+√(2-√2))
			//l=4 (C²-2)²-2)²-2 = (C-√(2+√(2+√2))) * (C+√(2+√(2+√2))) * (C-√(2-√(2+√2))) * (C+√(2-√(2+√2))) * (C-√(2+√(2-√2))) * (C+√(2+√(2-√2))) * (C-√(2-√(2-√2))) * (C+√(2-√(2-√2))) 
			for (int i = 0; i < 1 << l; i++)
			{
				T val = T.Zero;
				for (int j = 0; j < l; j++)
				{
					int sign = (((1 << j) & i) != 0) ? +1 : -1;
					val = T.Sqrt(val + T.CreateTruncating(2.0)) * T.CreateTruncating(sign);
				}
				action(i, val);
			}
		}

		public static int[] calculateWorkSizes(int maxVectors, int allWorkSize)
		{
			int cFullSteps = (allWorkSize + maxVectors - 1) / maxVectors;
			int[] lst = new int[cFullSteps];
			for (int i = 0; i < cFullSteps - 1; i++) lst[i] = maxVectors;
			lst[cFullSteps - 1] = allWorkSize - (cFullSteps - 1) * maxVectors;

			return lst;
		}

		public static int calculatePowOf2(int cSegments)
		{
			int log2 = (int)Math.Log(cSegments, 2);
			int newVal = 1 << log2;
			if (newVal < cSegments) log2++;
			return log2;
		}

		public static string getTypeName<T>()//for using inside OpenCL/CUDA programs
		{
			string strType = typeof(T).Name;

			if (strType == "Single") return "float";//can't return Single for OpenCL/CUDA programs
			if (strType == "Double") return "double";//can't return Double for OpenCL/CUDA programs
			if (strType == "DD128") return "DD128";
			if (strType == "QD256") return "QD256";

			return "float";
		}

		public static string getDataPath()
		{
			string strFoler = Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile);
			return Path.Combine(strFoler, "VLP2D_Data");
		}

		public static void addCustomFunctions<T>() where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>
		{
			ToolsHelper<T>.OperationsRegistry.addFunctionAfterRegistryInit("pi", ["pi"], 1, new CalculatorPi<T>());
		}
	}

	internal sealed class CalculatorPi<T> : IOperationCalculator<T> where T : INumber<T>, IFloatingPointConstants<T>
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public T calculate(Span<T> items, int idx) => T.Pi * (items[idx + 0]);
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Expression expression(Expression[] items, int idx) => Expression.Multiply(Expression.Constant(T.Pi, typeof(T)), items[idx + 0]);
	}
}

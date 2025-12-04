using System.IO;
using VLP2D.Common;
using VLP2D.Model;

namespace VLP2D.ViewModel
{
	internal class TaskParams
	{
		public bool isOpenCLCheckBoxChecked, isCUDACheckBoxChecked, isJordan, isChebysh, isBiconjugateStabilized, isVarSepProgonka;
		public int idxScheme, precision, paramL, idxInterpol, idxDeviceOCL, idxDeviceCUDA;
		public VarSepMethodsEnum varSepMethod;
		public CRMethodsEnum crMethod;

		public TaskParams()
		{
		}

		public static TaskParams restore(string fileName)
		{
			return UtilsJson.jsonDeserialize<TaskParams>(fullPath(fileName));
		}

		public void save(string fileName)
		{
			Directory.CreateDirectory(iniDirectory());
			UtilsJson.jsonSerialize<TaskParams>(this, fullPath(fileName));
		}

		static string iniDirectory() => Path.Combine(Utils.getDataPath(), "Ini");
		static string fullPath(string fileName) => Path.Combine(iniDirectory(), fileName + ".ini");
	}
}

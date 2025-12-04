#define AllowAbscentIterations

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using VLP2D.Model;
using VLP2D.Properties;

namespace VLP2D.ViewModel
{
	public class InputParser
	{
		HashSet<string> setKeys = new HashSet<string>();
		string[] keysRequired = { "xMax", "yMax", "segmentsX", "segmentsY", 
		};
		string keyEpsilon = "epsilon";
		string[] keysIterations = 
		{
			"SimpleIteration", "SlidingIteration", "SOR", "MultiGrid", 
			"Splitting", "VarDir", "PTM", "GradientDescent", "MinimumResidual",
			"ConjugateGradient", "BiconjugateGradient", "Chebishev3Layers", 
		};
		string[] keysTypeDependent;
		string[] keysTypeNames = { "Single", "Double", "DD128", "QD256", };
		string[] keysBoundaries1 = { "funcLeft", "funcRight", "funcTop", "funcBottom" };
		string[] keysBoundaries2 = { "funcBoundary" };

		string[] keysOptional = { "funcRHS", "funcAnalytic" };
		string[] keysOptionalOCL = { "SimpleIterationOCL", "SlidingIterationOCL", "SOROCL", "SplittingOCL", "VarDirOCL", "MultiGridOCL" };
		string[] keysOptionalCUDA = { "SimpleIterationCUDA", "SlidingIterationCUDA", "SORCUDA", "SplittingCUDA", "VarDirCUDA", "MultiGridCUDA" };

		public InputParser()
		{
			foreach (string key in keysRequired) setKeys.Add(key);
			foreach (string key in keysBoundaries1) setKeys.Add(key);
			foreach (string key in keysBoundaries2) setKeys.Add(key);

			foreach (string key in keysOptional) setKeys.Add(key);
			foreach (string key in keysOptionalOCL) setKeys.Add(key);
			foreach (string key in keysOptionalCUDA) setKeys.Add(key);

			keysTypeDependent = (new string[] { keyEpsilon }).ToList().Concat(keysIterations.ToList()).ToArray();
			addPrecisionKeys(keysTypeDependent);
			addPrecisionKeys(keysOptionalOCL);
			addPrecisionKeys(keysOptionalCUDA);
		}

		public VLPRectangleParams parseFile(string filePath)
		{
			VLPRectangleParams parsedParams = null;
			List<string> strErrors = new List<string>();
			Dictionary<string, string> dict = new Dictionary<string, string>();
			readParameters(filePath, dict, setKeys);
			bool reqPresent = parametersValuesPresent(dict, keysRequired, strErrors);
#if !AllowAbscentIterations
			bool reqPresent2 = parametersValuesPresent(dict, keysTypeDependent, keysTypeNames, strErrors);
#else
			bool reqPresent2 = true;
#endif
			bool altPresent = boundariesParametersValuesPresent(dict, keysBoundaries1, keysBoundaries2, strErrors);
			if (reqPresent && reqPresent2 && altPresent)
			{
				int segmentsX = parseInt(dict, "segmentsX", strErrors);
				int segmentsY = parseInt(dict, "segmentsY", strErrors);

				Dictionary<string, double> dictEpsilon = new Dictionary<string, double>();
				if (dict.ContainsKey(keyEpsilon))
				{
					double epsilon = parseDouble(dict, keyEpsilon, strErrors);
					dictEpsilon[keyEpsilon] = epsilon;
				}
				foreach (string keyPrecision in keysTypeNames)
				{
					string keyEps = keyEpsilon + keyPrecision;
					if (dict.ContainsKey(keyEps))
					{
						double eps = parseDouble(dict, keyEps, null);
						dictEpsilon[keyEps] = eps;
					}
				}

				Dictionary<string, int[]> itersCPU = new Dictionary<string, int[]>();
				fillMethodIterations(dict, itersCPU, typeof(SchemeCPUEnum), strErrors);

				Dictionary<string, int[]> itersOCL = new Dictionary<string, int[]>();
				fillMethodIterations(dict, itersOCL, typeof(SchemeOCLEnum), strErrors);

				Dictionary<string, int[]> itersCUDA = new Dictionary<string, int[]>();
				fillMethodIterations(dict, itersCUDA, typeof(SchemeCUDAEnum), strErrors);

				//take OCL iters values from CPU variants
				string[] keysCPU = { "SimpleIteration", "SlidingIteration", "SOR", "Splitting", "VarDir", "MultiGrid" };
				for (int i = 0; i < keysOptionalOCL.Length; i++)
				{
					if (itersOCL.GetValueOrDefault(keysOptionalOCL[i]) == null) itersOCL[keysOptionalOCL[i]] = itersCPU.GetValueOrDefault(keysCPU[i]);
				}
				//take CUDA iters values from CPU variants
				for (int i = 0; i < keysOptionalCUDA.Length; i++)
				{
					if (itersCUDA.GetValueOrDefault(keysOptionalCUDA[i]) == null) itersCUDA[keysOptionalCUDA[i]] = itersCPU.GetValueOrDefault(keysCPU[i]);
				}

				string replaceMaxes(string key) => dict.GetValueOrDefault(key)?.Replace("xMax", dict["xMax"]).Replace("yMax", dict["yMax"]);
				string fLeft = replaceMaxes("funcLeft");
				string fRight = replaceMaxes("funcRight");
				string fTop = replaceMaxes("funcTop");
				string fBottom = replaceMaxes("funcBottom");
				string RHS = replaceMaxes("funcRHS");
				string funcBoundary = replaceMaxes("funcBoundary");
				string funcAnalytic = replaceMaxes("funcAnalytic");

				if (strErrors.Count == 0)
				{
					string name = Path.GetFileNameWithoutExtension(filePath);
					parsedParams = new VLPRectangleParams(dict["xMax"], dict["yMax"], segmentsX, segmentsY, itersCPU, itersOCL, itersCUDA, dictEpsilon, RHS, fLeft, fRight, fTop, fBottom, funcBoundary, funcAnalytic, name);
				}
			}
			if (strErrors.Count > 0)
			{
				string fileName = String.Format(Resources.strFile, filePath);
				string allErrors = "";
				foreach (string msg in strErrors)
				{
					allErrors += msg + "\n";
				}
				MessageBox.Show(allErrors, fileName);
			}
			return parsedParams;
		}

		void readParameters(string path, Dictionary<string, string> dictParams, HashSet<string> keys)
		{
			using (StreamReader reader = new StreamReader(path))
			{
				string line;
				while ((line = reader.ReadLine()) != null)
				{
					int idx = line.IndexOf("=");
					if (idx != -1)
					{
						string key = line.Substring(0, idx);
						if (keys.Contains(key))//add only known keys
						{
							string val = line.Substring(idx + 1);
							if (val.Length > 0) dictParams[key] = val;//else will remains null
						}
					}
				}
			}
		}

		bool parametersValuesPresent(Dictionary<string, string> dictParams, string[] names, List<string> strErrors)
		{
			int count = 0;
			foreach (string name in names)
			{
				if (string.IsNullOrEmpty(dictParams.GetValueOrDefault(name)))
				{
					count++;
					strErrors.Add(string.Format(Resources.strReqParameterMissed, name));
				}
			}
			return count == 0;
		}

		bool parametersValuesPresent(Dictionary<string, string> dictParams, string[] names, string[] types, List<string> strErrors)
		{
			int count = 0;
			foreach (string name in names)
			{
				if (string.IsNullOrEmpty(dictParams.GetValueOrDefault(name)))
				{
					int cPresent = 0;
					foreach (string typeName in types)
					{
						if (!string.IsNullOrEmpty(dictParams.GetValueOrDefault(name + typeName))) cPresent++;
					}
					if (cPresent != types.Length)
					{
						count++;
						strErrors.Add(string.Format(Resources.strReqParameterMissed, name));
					}
				}
			}
			return count == 0;
		}

		bool boundariesParametersValuesPresent(Dictionary<string, string> dictParams, string[] names1, string[] names2, List<string> strErrors)
		{
			int count = 0;
			foreach (string name in names1) if (!string.IsNullOrEmpty(dictParams.GetValueOrDefault(name))) count++;
			if (count == names1.Length) return true;

			count = 0;
			foreach (string name in names2) if (!string.IsNullOrEmpty(dictParams.GetValueOrDefault(name))) count++;
			if (count == names2.Length) return true;

			strErrors.Add(Resources.strBoundaryParameters);

			return false;
		}

		int parseInt(Dictionary<string, string> dict, string paramName, List<string> strErrors)
		{
			int rc = 0;
			try
			{
				rc = int.Parse(dict.GetValueOrDefault(paramName));
			}
			catch (Exception ex)
			{
				string strErr = string.Format(Resources.strParsingParameterError, paramName, dict.GetValueOrDefault(paramName));
				strErrors.Add(strErr + "\n\n" + ex.Message);
			}

			return rc;
		}

		double parseDouble(Dictionary<string, string> dict, string paramName, List<string> strErrors)
		{
			double rc = 0.0;
			try
			{
				rc = double.Parse(dict.GetValueOrDefault(paramName), System.Globalization.CultureInfo.InvariantCulture);
			}
			catch (Exception ex)
			{
				if (strErrors != null)
				{
					string strErr = string.Format(Resources.strParsingParameterError, paramName, dict.GetValueOrDefault(paramName));
					strErrors.Add(strErr + "\n\n" + ex.Message);
				}
			}

			return rc;
		}

		int[] parseMethodIterations(Dictionary<string, string> dict, string paramName, List<string> strErrors)
		{
			int[] rc = null;
			try
			{
				string val = dict.GetValueOrDefault(paramName);
				if (val != null)
				{
					rc = val.Split(',').Select(int.Parse).ToArray();
					if (rc != null && rc.Length < 4)
					{
						string strErr = string.Format(Resources.strParsingParameterError, paramName, dict.GetValueOrDefault(paramName));
						strErrors.Add(strErr + "\n\n" + Resources.strFour);
					}
				}
			}
			catch (Exception ex)
			{
				string strErr = string.Format(Resources.strParsingParameterError, paramName, dict.GetValueOrDefault(paramName));
				strErrors.Add(strErr + "\n\n" + ex.Message);
			}
			return rc;
		}

		void fillMethodIterations(Dictionary<string, string> dict, Dictionary<string, int[]> dictIters, Type enumType, List<string> strErrors)
		{
			foreach (string enumName in Enum.GetNames(enumType))
			{
				if (dict.ContainsKey(enumName))
				{
					int[] itersVal = parseMethodIterations(dict, enumName, strErrors);
					if (itersVal != null) dictIters.Add(enumName, itersVal);
				}
#if AllowAbscentIterations
				else dictIters.Add(enumName, [5000, 5000, 5000, 5000]);
#endif
				foreach (string keyPrecision in keysTypeNames)
				{
					string keyIter = enumName + keyPrecision;
					if (dict.ContainsKey(keyIter))
					{
						int[] itersVal = parseMethodIterations(dict, keyIter, strErrors);
						if (itersVal != null) dictIters.Add(keyIter, itersVal);
					}
#if AllowAbscentIterations
					else dictIters.Add(keyIter, [5000, 5000, 5000, 5000]);
#endif
				}
			}
		}

		void addPrecisionKeys(string[] keys)
		{
			foreach (string key in keys)
			{
				setKeys.Add(key);
				foreach (string keyPrecision in keysTypeNames) setKeys.Add(key + keyPrecision);
			}
		}
	}
}

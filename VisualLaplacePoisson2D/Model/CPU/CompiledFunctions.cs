using ELW.Library.Math.Tools;
using System;
using System.Numerics;
using VLP2D.Common;
using VLP2D.Properties;

namespace VLP2D.Model
{
	public class CompiledFunctions<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>
	{
		readonly Calculator1D<T> calculatorLeft, calculatorRight, calculatorTop, calculatorBottom;
		readonly Calculator2D<T> calculatorKsi, calculatorBoundary, calculatorAnalytic;
		string strErrors = "";

		public Func<T, T> pFuncLeft
		{
			get { return calculatorLeft != null ? calculatorLeft.Calculate : (Func<T, T>)null; }
		}

		public Func<T, T> pFuncRight
		{
			get { return calculatorRight != null ? calculatorRight.Calculate : (Func<T, T>)null; }
		}

		public Func<T, T> pFuncTop
		{
			get { return calculatorTop != null ? calculatorTop.Calculate : (Func<T, T>)null; }
		}

		public Func<T, T> pFuncBottom
		{
			get { return calculatorBottom != null ? calculatorBottom.Calculate : (Func<T, T>)null; }
		}

		public Func<T, T, T> pFuncKsi
		{
			get { return calculatorKsi != null ? calculatorKsi.Calculate : (Func<T, T, T>)null; }
		}

		public Func<T, T, T> pFuncBoundary
		{
			get { return calculatorBoundary != null ? calculatorBoundary.Calculate : (Func<T, T, T>)null; }
		}

		public Func<T, T, T> pFuncAnalytic
		{
			get { return calculatorAnalytic != null ? calculatorAnalytic.Calculate : (Func<T, T, T>)null; }
		}

		public CompiledFunctions(string strFuncLeft, string strFuncRight, string strFuncTop, string strFuncBottom, string strFuncKsi, string strFuncBoundary, string strFuncAnalytic, bool bCompile = false)
		{
			Utils.addCustomFunctions<T>();

			string strErr;

			calculatorLeft = createCalculator1(strFuncLeft, "funcLeft ");
			calculatorRight = createCalculator1(strFuncRight, "funcRight ");
			calculatorTop = createCalculator1(strFuncTop, "funcTop ");
			calculatorBottom = createCalculator1(strFuncBottom, "funcBottom ");

			calculatorBoundary = createCalculator2(strFuncBoundary, "funcBoundary ");
			calculatorKsi = createCalculator2(strFuncKsi, "funcRHS ");
			calculatorAnalytic = createCalculator2(strFuncAnalytic, "funcAnalytic ");

			if (strErrors.Length > 0) System.Windows.MessageBox.Show(strErrors, Resources.strExpressionError);

			Calculator1D<T> createCalculator1(string strFunc, string strPrefix)
			{
				Calculator1D<T> calc = null;
				if (!string.IsNullOrEmpty(strFunc))
				{
					calc = new Calculator1D<T>();
					strErr = calc.setExpression(strFunc, bCompile);
					if (strErr != null) strErrors += createErrorString(strPrefix, strFunc, strErr);
				}
				return calc;
			}

			Calculator2D<T> createCalculator2(string strFunc, string strPrefix)
			{
				Calculator2D<T> calc = null;
				if (!string.IsNullOrEmpty(strFunc))
				{
					calc = new Calculator2D<T>();
					strErr = calc.setExpression(strFunc, "x", bCompile);
					if (strErr != null) strErrors += createErrorString(strPrefix, strFunc, strErr);
				}
				return calc;
			}
		}

		string createErrorString(string paramName, string strFunc, string strErr)
		{
			return paramName + "'" + strFunc + "'" + " : " + strErr + "\n";
		}

		public string stringErrors()
		{
			return strErrors.Length > 0 ? strErrors : null;
		}
	}
}

using ELW.Library.Math;
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

		public CompiledFunctions(TaskInputParams pParams, bool bCompile = false)
		{
			if (pParams == null) return;
			Utils.addCustomFunctions<T>();

			string strErr;

			calculatorLeft = createCalculator1(pParams.funcLeft, "funcLeft ");
			calculatorRight = createCalculator1(pParams.funcRight, "funcRight ");
			calculatorTop = createCalculator1(pParams.funcTop, "funcTop ");
			calculatorBottom = createCalculator1(pParams.funcBottom, "funcBottom ");

			calculatorBoundary = createCalculator2(pParams.funcBoundary, "funcBoundary ");
			calculatorKsi = createCalculator2(pParams.fKsi, "funcRHS ");

			if (pParams.series1Function != null)
			{
				try
				{
					CalculatorSumIXY<T> calcSum1 = new CalculatorSumIXY<T>(pParams.series1IndexMin, pParams.series1IndexMax, pParams.series1Step, pParams.series1Multiplier, pParams.series1Function, bCompile, GridIterator.optionsParallel);
					ToolsHelper<T>.OperationsRegistry.addFunctionAfterRegistryInit("sumIXY", ["sumIXY"], 2, calcSum1);
				}
				catch (Exception ex)
				{
					strErrors += ex.Message + "\n";
				}
			}

			if (pParams.series2Function != null)
			{
				try
				{
					CalculatorSumIJXY<T> calcSum2 = new CalculatorSumIJXY<T>(pParams.series2IndexMin, pParams.series2IndexMax, pParams.series2Step, pParams.series2Multiplier, pParams.series2Function, bCompile, GridIterator.optionsParallel);
					ToolsHelper<T>.OperationsRegistry.addFunctionAfterRegistryInit("sumIJXY", ["sumIJXY"], 2, calcSum2);
				}
				catch (Exception ex)
				{
					strErrors += ex.Message + "\n";
				}
			}
			calculatorAnalytic = createCalculator2(pParams.funcAnalytic, "funcAnalytic ");//call after adding sum1, sum2

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

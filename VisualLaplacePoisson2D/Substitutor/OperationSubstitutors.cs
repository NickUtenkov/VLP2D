using System;

namespace MathSubstitutor
{
	public interface IOperationSubstitutor
	{
		/// <summary>
		/// Returns a result of operation called.
		/// </summary>
		RPNItem substitute(Span<RPNItem> parameters, int idx);
	}

	internal sealed class OperatorAddition : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx) => Oper.operatorAsVariable("add", parameters, idx);
	}

	internal sealed class OperatorSubtraction : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx) => Oper.operatorAsVariable("sub", parameters, idx);
	}

	internal sealed class OperatorMultiplication : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx) => Oper.operatorAsVariable("mul", parameters, idx);
	}

	internal sealed class OperatorDivision : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx) => Oper.operatorAsVariable("div", parameters, idx);
	}

	internal sealed class OperatorPositivation : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx) => Oper.positivationAsVariable(parameters, idx);
	}

	internal sealed class OperatorNegation : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx) => Oper.negationAsVariable(parameters, idx);
	}

	internal sealed class OperatorPowering : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx)
		{
			return null;
		}
	}

	internal sealed class OperatorConditional : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx)
		{
			return null;
		}
	}

	public static class Oper
	{
		public static RPNItem operatorAsVariable(string prefix, Span<RPNItem> operationsStack, int idx)
		{
			string str = "";
			if (operationsStack[idx + 0].kind == RPNItemKind.Variable && operationsStack[idx + 1].kind == RPNItemKind.Variable)
			{
				str = string.Format(prefix + "_HH({0}, {1})", operationsStack[idx + 0].variableOrConstant, operationsStack[idx + 1].variableOrConstant);
			}
			if (operationsStack[idx + 0].kind == RPNItemKind.Variable && operationsStack[idx + 1].kind == RPNItemKind.Constant)
			{
				str = string.Format(prefix + "_HD({0}, {1})", operationsStack[idx + 0].variableOrConstant, operationsStack[idx + 1].variableOrConstant);
			}
			if (operationsStack[idx + 0].kind == RPNItemKind.Constant && operationsStack[idx + 1].kind == RPNItemKind.Variable)
			{
				str = string.Format(prefix + "_DH({0}, {1})", operationsStack[idx + 0].variableOrConstant, operationsStack[idx + 1].variableOrConstant);
			}
			return new RPNItem(RPNItemKind.Variable, str);
		}

		public static RPNItem positivationAsVariable(Span<RPNItem> operationsStack, int idx)
		{
			string str = "";
			if (operationsStack[idx + 0].kind == RPNItemKind.Variable)
			{
				str = string.Format("positive({0})", operationsStack[idx + 0].variableOrConstant);
			}
			if (operationsStack[idx + 0].kind == RPNItemKind.Constant)
			{
				str = string.Format("fabs({0})", operationsStack[idx + 0].variableOrConstant);
			}
			return new RPNItem(RPNItemKind.Variable, str);
		}

		public static RPNItem negationAsVariable(Span<RPNItem> operationsStack, int idx)
		{
			string str = "";
			if (operationsStack[idx + 0].kind == RPNItemKind.Variable)
			{
				str = string.Format("negative({0})", operationsStack[idx + 0].variableOrConstant);
			}
			if (operationsStack[idx + 0].kind == RPNItemKind.Constant)
			{
				str = string.Format("-{0}", operationsStack[idx + 0].variableOrConstant);
			}
			return new RPNItem(RPNItemKind.Variable, str);
		}
	}
}
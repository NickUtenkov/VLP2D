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
			RPNItem arg1 = parameters[idx + 0];
			RPNItem arg2 = parameters[idx + 1];
			string str = string.Format("pow({0},{1})", arg1.variableOrConstant, arg2.variableOrConstant);
			return new RPNItem(RPNItemKind.Variable, str);
		}
	}

	internal sealed class OperatorConditional : IOperationSubstitutor
	{
		public RPNItem substitute(Span<RPNItem> parameters, int idx)
		{
			RPNItem arg1 = parameters[idx + 0];
			RPNItem arg2 = parameters[idx + 1];
			RPNItem arg3 = parameters[idx + 2];
			string str = string.Format("{0}?{1}:{2}", arg1.variableOrConstant, arg2.variableOrConstant, arg3.variableOrConstant);
			return new RPNItem(RPNItemKind.Variable, str);
		}
	}

	public static class Oper
	{
		public static RPNItem operatorAsVariable(string prefix, Span<RPNItem> operationsStack, int idx)
		{
			string str = "";
			RPNItem arg1 = operationsStack[idx + 0];
			RPNItem arg2 = operationsStack[idx + 1];
			if (arg1.kind == RPNItemKind.Variable && arg2.kind == RPNItemKind.Variable)
			{
				str = string.Format(prefix + "_HH({0}, {1})", arg1.variableOrConstant, arg2.variableOrConstant);
			}
			if (arg1.kind == RPNItemKind.Variable && arg2.kind == RPNItemKind.Constant)
			{
				str = string.Format(prefix + "_HD({0}, {1})", arg1.variableOrConstant, arg2.variableOrConstant);
			}
			if (arg1.kind == RPNItemKind.Constant && arg2.kind == RPNItemKind.Variable)
			{
				str = string.Format(prefix + "_DH({0}, {1})", arg1.variableOrConstant, arg2.variableOrConstant);
			}
			return new RPNItem(RPNItemKind.Variable, str);
		}

		public static RPNItem positivationAsVariable(Span<RPNItem> operationsStack, int idx)
		{
			string str = "";
			RPNItem arg1 = operationsStack[idx + 0];
			if (arg1.kind == RPNItemKind.Variable)
			{
				str = string.Format("positive({0})", arg1.variableOrConstant);
			}
			if (arg1.kind == RPNItemKind.Constant)
			{
				str = string.Format("fabs({0})", arg1.variableOrConstant);
			}
			return new RPNItem(RPNItemKind.Variable, str);
		}

		public static RPNItem negationAsVariable(Span<RPNItem> operationsStack, int idx)
		{
			string str = "";
			RPNItem arg1 = operationsStack[idx + 0];
			if (arg1.kind == RPNItemKind.Variable)
			{
				str = string.Format("negative({0})", arg1.variableOrConstant);
			}
			if (arg1.kind == RPNItemKind.Constant)
			{
				str = string.Format("-{0}", arg1.variableOrConstant);
			}
			return new RPNItem(RPNItemKind.Variable, str);
		}
	}
}
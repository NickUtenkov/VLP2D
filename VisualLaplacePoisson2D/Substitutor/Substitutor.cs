using System;

namespace MathSubstitutor
{
	/// <summary>
	/// Quick access to tool wanted.
	/// </summary>
	public static class Substitutor
	{
		public static OperationsRegistry operationsRegistry { get; private set; }
		public static Parser parser { get; private set; }

		public static Compiler compiler { get; private set; }

		static Substitutor()
		{
			operationsRegistry = new OperationsRegistry();

			parser = new Parser(operationsRegistry);
			compiler = new Compiler(operationsRegistry);
		}

		public static string replaceArithmeticOperators(string strInput)
		{
			string strNoSpaces = strInput.Replace(" ", "");
			string strErr;
			RPNList compiledExpression = createCompiledExpression(strNoSpaces, out strErr);
			return createDecompiledString(compiledExpression);
		}

		static RPNList createCompiledExpression(string strFunc, out string errStr)
		{
			errStr = null;
			if (strFunc == null || strFunc.Length == 0) return null;

			RPNList expr = null;
			try
			{
				PreparedExpression preparedExpression = parser.Parse(strFunc);
				expr = compiler.Compile(preparedExpression);
			}
			/*catch (CompilerSyntaxException ex)
			{
				errStr = ex.Message;
			}
			catch (MathProcessorException ex)
			{
				errStr = ex.Message;
			}*/
			catch (ArgumentException ex)
			{
				errStr = ex.Message;
			}
			catch (Exception ex)
			{
				errStr = ex.Message;
			}

			return expr;
		}

		static string createDecompiledString(RPNList compiledExpression)
		{
			RPNItem[] calculationsStack = new RPNItem[compiledExpression.RPNItems.Count];
			int idx = -1;//idx + 1 == length(count)

			for (int i = 0; i < compiledExpression.RPNItems.Count; i++)
			{
				RPNItem item = compiledExpression.RPNItems[i];

				switch (item.kind)
				{
					case RPNItemKind.Constant:
					case RPNItemKind.Variable:
						idx++;
						calculationsStack[idx] = item;
						break;
					case RPNItemKind.Operation:
						Operation operation = item.operation;

						idx -= operation.operandsCount;
						idx++;
						calculationsStack[idx] = operation.substitutor.substitute(calculationsStack, idx);
						break;
					default:
						break;
				}
			}

			return calculationsStack[0].variableOrConstant;
		}
	}
}
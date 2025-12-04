using System;
using System.Collections.Generic;

namespace MathSubstitutor
{
	/// <summary>
	/// Implements compiler logic.
	/// </summary>
	public sealed class Compiler
	{
		public OperationsRegistry operationsRegistry { get; private set; }

		public Compiler(OperationsRegistry operationsRegistry)
		{
			if (operationsRegistry == null) throw new ArgumentNullException("operationsRegistry");

			this.operationsRegistry = operationsRegistry;
		}

		/// <summary>
		/// Returns a compiled expression for specified source string.
		/// </summary>
		public RPNList Compile(PreparedExpression preparedExpression)
		{
			if (preparedExpression == null) throw new ArgumentNullException("preparedExpression");

			OperationsStack operationsStack = new OperationsStack(operationsRegistry);

			for (int itemIndex = 0; itemIndex < preparedExpression.PreparedExpressionItems.Count; itemIndex++)
			{
				PreparedExpressionItem item = preparedExpression.PreparedExpressionItems[itemIndex];
				// If constant or variable - add to result
				if (item.kind == PreparedExpressionItemKind.Constant) operationsStack.PushConstant(item.variableOrConstant);
				if (item.kind == PreparedExpressionItemKind.Variable) operationsStack.PushVariable(item.variableOrConstant);
				// If delimiter
				if (item.kind == PreparedExpressionItemKind.Delimiter) operationsStack.PushDelimiter(item.delimiterKind);
				// Signature (operator signature / part of signature / function)
				if (item.kind == PreparedExpressionItemKind.Signature)
				{
					List<Operation> operations = new List<Operation>(operationsRegistry.GetOperationsUsingSignature(item.signature));
					operations.Sort(new Comparison<Operation>(compareOperationsByOperandsCount));

					for (int i = 0; i < operations.Count; i++)
					{
						Operation operation = operations[i];
						// Operator
						if (operation.kind == OperationKind.Operator)
						{
							// Unary operator
							if (operation.operandsCount == 1)
							{
								// If operator placed at the start of subexpression
								if ((itemIndex == 0) || ((itemIndex > 0) && 
									(preparedExpression.PreparedExpressionItems[itemIndex - 1].kind == PreparedExpressionItemKind.Delimiter) && 
									(preparedExpression.PreparedExpressionItems[itemIndex - 1].delimiterKind == DelimiterKind.OpeningBrace)))
								{
									operationsStack.PushUnaryOperator(operation);
									break;
								}
							}
							// Binary operator
							if (operation.operandsCount == 2)
							{
								operationsStack.PushBinaryOperator(operation);
								break;
							}
							// Ternary and more
							if (operation.operandsCount > 2)
							{
								int partNumber = 0;
								for (int k = 0; k < operation.signature.Length; k++)
								{
									if (operation.signature[k] == item.signature)
									{
										partNumber = k + 1;
										break;
									}
								}
								// If it is start part in signature
								if (partNumber == 1)
								{
									operationsStack.PushComplexOperatorFirstSignature(operation);
									break;
								}

								operationsStack.PushComplexOperatorNonFirstSignature(operation, partNumber);
								break;
							}
						}
						// Function
						/*if (operation.kind == OperationKind.Function)
						{
							operationsStack.PushFunction(operation);
							break;
						}*/
					}
				}
			}

			operationsStack.DoFinalFlush();

			RPNList res = operationsStack.GetResult();
			if (!isCompiledExpressionStackBalanced(res)) throw new Exception("Operands disbalance detected.");
			return res;
		}

		/// <summary>
		/// Comparison.
		/// </summary>
		private static int compareOperationsByOperandsCount(Operation x, Operation y)
		{
			/*if (x == null)
			{
				if (y == null) return 0;
				return -1;
			}
			if (y == null) return 1;*/
			//
			if (x.operandsCount > y.operandsCount) return 1;
			if (x.operandsCount < y.operandsCount) return -1;
			return 0;
		}

		/// <summary>
		/// Checks a compiled expression for stack balance.
		/// </summary>
		private bool isCompiledExpressionStackBalanced(RPNList compiledExpression)
		{
			if (compiledExpression == null) throw new ArgumentNullException("compiledExpression");

			int stackPointer = 0;

			for (int i = 0; i < compiledExpression.RPNItems.Count; i++)
			{
				RPNItem item = compiledExpression.RPNItems[i];

				switch (item.kind)
				{
					case RPNItemKind.Constant:
						stackPointer++;
						break;
					case RPNItemKind.Variable:
						stackPointer++;
						break;
					case RPNItemKind.Operation:
						Operation operation = item.operation;
						stackPointer -= operation.operandsCount - 1;
						break;
					default:
						throw new InvalidOperationException("Unknown item kind.");
				}
			}

			if (stackPointer != 1) return false;
			return true;
		}

		#region Nested

		/// <summary>
		/// Manages stack structure, forms a result sequence.
		/// </summary>
		private sealed class OperationsStack
		{
			private readonly List<RPNItem> res = new List<RPNItem>();
			private readonly List<OperationsStackItem> stack = new List<OperationsStackItem>();
			private readonly OperationsRegistry operationsRegistry;

			public OperationsStack(OperationsRegistry operationsRegistry)
			{
				if (operationsRegistry == null) throw new ArgumentNullException("operationsRegistry");
				this.operationsRegistry = operationsRegistry;
			}

			public void PushConstant(string constant) => res.Add(new RPNItem(RPNItemKind.Constant, constant));
			public void PushVariable(string variableName) => res.Add(new RPNItem(RPNItemKind.Variable, variableName));

			public void PushDelimiter(DelimiterKind delimiterKind)
			{
				if (delimiterKind == DelimiterKind.OpeningBrace)
				{
					stack.Add(new OperationsStackItem(OperationStackItemKind.Delimiter, DelimiterKind.OpeningBrace));
				}
				if (delimiterKind == DelimiterKind.ClosingBrace)
				{
					// Pop all items before previous OpeningBrace includes it
					int j = stack.Count - 1;
					while (j >= 0)
					{
						if ((stack[j].Kind == OperationStackItemKind.Delimiter) && (stack[j].Delimiter == DelimiterKind.OpeningBrace))
						{
							stack.RemoveAt(j);
							break;
						}
						//
						switch (stack[j].Kind)
						{
							case OperationStackItemKind.Operation:
								Operation operation = operationsRegistry.GetOperationByName(stack[j].OperationName);
								res.Add(new RPNItem(RPNItemKind.Operation, operation));
								stack.RemoveAt(j);
								break;
							default:
								throw new Exception("Unexpected item in stack.");
						}
						j--;
					}
					if (j < 0) throw new Exception("Braces syntax error.");
					// If previous item is function - pop it
					if (stack.Count > 0)
					{
						if (stack[stack.Count - 1].Kind == OperationStackItemKind.Operation)
						{
							Operation operation = operationsRegistry.GetOperationByName(stack[stack.Count - 1].OperationName);
							/*if (operation.kind == OperationKind.Function)
							{
								res.Add(new RPNItem(RPNItemKind.Operation, operation));
								stack.RemoveAt(stack.Count - 1);
							}*/
						}
					}
				}
				if (delimiterKind == DelimiterKind.Comma)
				{
					// Pop all items before previous OpeningBrace excludes it
					int j = stack.Count - 1;
					while (j >= 0)
					{
						if ((stack[j].Kind == OperationStackItemKind.Delimiter) && (stack[j].Delimiter == DelimiterKind.OpeningBrace)) break;

						switch (stack[j].Kind)
						{
							case OperationStackItemKind.Operation:
								Operation operation = operationsRegistry.GetOperationByName(stack[j].OperationName);
								res.Add(new RPNItem(RPNItemKind.Operation, operation));
								stack.RemoveAt(j);
								break;
							default:
								throw new Exception("Unexpected item in stack.");
						}
						j--;
					}
					if (j < 0) throw new Exception("Braces syntax error.");
				}
			}

			private void pushOperationAccordingToAssociationAndPriority(Operation operation, OperationsStackItem itemToPush)
			{
				// Push an operation according to association and priority
				if (stack.Count > 0)
				{
					int j = stack.Count - 1;
					bool priorityExit = false;
					while ((j >= 0) && (!priorityExit))
					{
						if (stack[j].Kind == OperationStackItemKind.Delimiter) break;

						switch (stack[j].Kind)
						{
							case OperationStackItemKind.Operation:
								if (operationsRegistry.GetAssociationByPriority(operation.priority) == PriorityAssociation.LeftAssociated)
								{
									if (operationsRegistry.GetOperationByName(stack[j].OperationName).priority > operation.priority) priorityExit = true;
								}
								else
								{
									if (operationsRegistry.GetOperationByName(stack[j].OperationName).priority >= operation.priority) priorityExit = true;
								}
								if (!priorityExit)
								{
									Operation operation1 = operationsRegistry.GetOperationByName(stack[j].OperationName);
									res.Add(new RPNItem(RPNItemKind.Operation, operation1));
									stack.RemoveAt(j);
								}
								break;
							case OperationStackItemKind.PartialSignature:
								priorityExit = true;
								break;
							default:
								throw new Exception("Unexpected item in stack.");
						}
						j--;
					}
				}
				stack.Add(itemToPush);
			}

			public void PushUnaryOperator(Operation operation)
			{
				pushOperationAccordingToAssociationAndPriority(operation,
					new OperationsStackItem(OperationStackItemKind.Operation, operation.name));
			}

			public void PushBinaryOperator(Operation operation)
			{
				pushOperationAccordingToAssociationAndPriority(operation,
					new OperationsStackItem(OperationStackItemKind.Operation, operation.name));
			}

			public void PushFunction(Operation operation) => stack.Add(new OperationsStackItem(OperationStackItemKind.Operation, operation.name));

			public void PushComplexOperatorFirstSignature(Operation operation)
			{
				PartialSignature signature = new PartialSignature();
				signature.operationName = operation.name;
				signature.signaturePartNumber = 1;

				pushOperationAccordingToAssociationAndPriority(operation,
					new OperationsStackItem(OperationStackItemKind.PartialSignature, signature));
			}

			public void PushComplexOperatorNonFirstSignature(Operation operation, int partNumber)
			{
				int j = stack.Count - 1;
				while (j >= 0)
				{
					if ((stack[j].Kind == OperationStackItemKind.PartialSignature) && (stack[j].PartialSignature.operationName == operation.name) && 
						(stack[j].PartialSignature.signaturePartNumber == partNumber - 1)) break;

					switch (stack[j].Kind)
					{
						case OperationStackItemKind.Operation:
							Operation operation1 = operationsRegistry.GetOperationByName(stack[j].OperationName);
							res.Add(new RPNItem(RPNItemKind.Operation, operation1));
							stack.RemoveAt(j);
							break;
						default:
							throw new Exception("Unexpected item in stack.");
					}
					j--;
				}
				if (j < 0) throw new Exception("Braces syntax error.");
				PartialSignature signature = new PartialSignature();
				signature.operationName = operation.name;
				signature.signaturePartNumber = partNumber;

				stack.Add(new OperationsStackItem(OperationStackItemKind.PartialSignature, signature));

				if (partNumber == operation.signature.Length)
				{
					for (int ii = 0; ii < partNumber; ii++) stack.RemoveAt(stack.Count - 1);
					stack.Add(new OperationsStackItem(OperationStackItemKind.Operation, operation.name));
				}
			}

			public void DoFinalFlush()
			{
				// Pop all from the stack
				for (int j = stack.Count - 1; j >= 0; j--)
				{
					switch (stack[j].Kind)
					{
						case OperationStackItemKind.Operation:
							Operation operation = operationsRegistry.GetOperationByName(stack[j].OperationName);
							res.Add(new RPNItem(RPNItemKind.Operation, operation));
							break;
						default:
							throw new Exception("Syntax error. Unexpected item in stack.");
					}
				}
			}

			public RPNList GetResult() => new RPNList(res);
		}

		private sealed class OperationsStackItem
		{
			public OperationStackItemKind Kind { get; private set; }

			private readonly DelimiterKind delimiter;
			public DelimiterKind Delimiter
			{
				get
				{
					if (Kind != OperationStackItemKind.Delimiter) throw new InvalidOperationException("Type mismatch.");
					return delimiter;
				}
			}

			private readonly string operationName;
			public string OperationName
			{
				get
				{
					if (Kind != OperationStackItemKind.Operation) throw new InvalidOperationException("Type mismatch.");
					return operationName;
				}
			}

			private readonly PartialSignature partialSignature;
			public PartialSignature PartialSignature
			{
				get
				{
					if (Kind != OperationStackItemKind.PartialSignature) throw new InvalidOperationException("Type mismatch.");
					return partialSignature;
				}
			}

			public OperationsStackItem(OperationStackItemKind kind, object value)
			{
				if (value == null) throw new ArgumentNullException("value");

				this.Kind = kind;
				switch (kind)
				{
					case OperationStackItemKind.Delimiter:
						delimiter = (DelimiterKind) value;
						break;
					case OperationStackItemKind.Operation:
						operationName = (string) value;
						break;
					case OperationStackItemKind.PartialSignature:
						partialSignature = (PartialSignature) value;
						break;
					default:
						throw new InvalidOperationException("Unexpected item kind.");
				}
			}
		}

		private struct PartialSignature
		{
			public string operationName { get; set; }
			public int signaturePartNumber { get; set; }
		}

		private enum OperationStackItemKind
		{
			Delimiter = 1,
			Operation = 2,
			PartialSignature = 3
		}

		#endregion
	}
}
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;

namespace MathSubstitutor
{
	public sealed class Parser
	{
		public OperationsRegistry operationsRegistry { get; private set; }

		public Parser(OperationsRegistry operationsRegistry)
		{
			if (operationsRegistry == null) throw new ArgumentNullException("operationsRegistry");

			this.operationsRegistry = operationsRegistry;
		}

		/// <summary>
		/// Prepares source string for compilation.
		/// </summary>
		public PreparedExpression Parse(string sourceString)
		{
			if (sourceString == null) throw new ArgumentNullException("sourceString");
			if (sourceString.Length == 0) throw new ArgumentException("String is empty.", "sourceString");
			// Signatures lenghts
			int[] lens = operationsRegistry.SignaturesLens;

			List<PreparedExpressionItem> res = new List<PreparedExpressionItem>();
			bool operandStarted = false;
			int operandStartIndex = 0;

			for (int i = 0; i < sourceString.Length; i++)
			{
				PreparedExpressionItem additionalItem = new PreparedExpressionItem(PreparedExpressionItemKind.Constant, "");//for not compiler error
				bool itemCreated = false;
				// Check for delimiters
				if ((sourceString[i] == '(') || (sourceString[i] == ')') || (sourceString[i] == ','))
				{
					// Storing delimiter
					DelimiterKind delimiterKind = new DelimiterKind();
					switch (sourceString[i])
					{
						case '(':
							delimiterKind = DelimiterKind.OpeningBrace;
							break;
						case ')':
							delimiterKind = DelimiterKind.ClosingBrace;
							break;
						case ',':
							delimiterKind = DelimiterKind.Comma;
							break;
					}
					additionalItem = new PreparedExpressionItem(PreparedExpressionItemKind.Delimiter, delimiterKind);
					itemCreated = true;
				}
				// If not found, check for signatures, from max length to min
				if (!itemCreated)
				{
					for (int j = lens.Length - 1; j >= 0; j--)
					{
						if (i + lens[j] <= sourceString.Length)
						{
							// If signature found
							if (operationsRegistry.IsSignatureDefined(sourceString.Substring(i, lens[j])))
							{
								// Storing signature
								additionalItem = new PreparedExpressionItem(PreparedExpressionItemKind.Signature, sourceString.Substring(i, lens[j]));
								itemCreated = true;
								break;
							}
						}
					}
				}
				// If not found, working with operand
				if (!itemCreated)
				{
					if (!operandStarted)
					{
						operandStarted = true;
						operandStartIndex = i;
					}
				}
				else
				{
					// NOTE: Duplicate code
					// Storing operand (constant or variable)
					if (operandStarted)
					{
						string operandString = sourceString.Substring(operandStartIndex, i - operandStartIndex);
						//Debug.WriteLine(string.Format("operandString1 '{0}'", operandString));
						if (double.TryParse(operandString, CultureInfo.InvariantCulture, out _))
						{
							//Debug.WriteLine(string.Format("parsed constant1 {0}", operandString));
							res.Add(new PreparedExpressionItem(PreparedExpressionItemKind.Constant, operandString));
						}
						else
						{
							if (!IsValidVariableName(operandString))
								throw new Exception(String.Format("{0} is not valid variable identifier1.", operandString));
							//
							res.Add(new PreparedExpressionItem(PreparedExpressionItemKind.Variable, operandString));
						}
						operandStarted = false;
					}
					// Delayed storing a delimiter or signature
					res.Add(additionalItem);
					// If additionalItem was a signature, we should add correct i index according to signature lenght
					if (additionalItem.kind == PreparedExpressionItemKind.Signature) i += additionalItem.signature.Length - 1;
				}
			}
			// Storing operand (constant or variable)
			if (operandStarted)
			{
				string operandString = sourceString.Substring(operandStartIndex);
				//Debug.WriteLine(string.Format("operandString2 '{0}'", operandString));
				if (double.TryParse(operandString, CultureInfo.InvariantCulture, out _))
				{
					//Debug.WriteLine(string.Format("parsed constant2 {0}", operandString));
					res.Add(new PreparedExpressionItem(PreparedExpressionItemKind.Constant, operandString));
				}
				else
				{
					if (!IsValidVariableName(operandString))
						throw new Exception(String.Format("{0} is not valid variable identifier2.", operandString));

					res.Add(new PreparedExpressionItem(PreparedExpressionItemKind.Variable, operandString));
				}
			}

			return new PreparedExpression(res);
		}

		public static bool IsValidVariableName(string @string)
		{
			if (@string == null) throw new ArgumentNullException("string");
			if (@string.Length == 0) return false;// Empty strings are not allowed
			if (!(Char.IsLetter(@string[0]) || @string[0] == '_')) return false;// Variable must be started from letter
			foreach (char c in @string) if (!(Char.IsLetterOrDigit(c) || c == '_')) return false;// All symbols must be letter or digit

			return true;
		}
	}
}
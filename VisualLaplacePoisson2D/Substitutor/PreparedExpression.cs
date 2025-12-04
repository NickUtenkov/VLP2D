using System;
using System.Collections.Generic;

namespace MathSubstitutor
{
	/// <summary>
	/// Delimiters supported.
	/// </summary>
	public enum DelimiterKind
	{
		OpeningBrace = 1,
		ClosingBrace = 2,
		Comma = 3
	}

	/// <summary>
	/// Type of prepared expression item content.
	/// </summary>
	public enum PreparedExpressionItemKind
	{
		/// <summary>
		/// Constant value.
		/// </summary>
		Constant,
		/// <summary>
		/// Variable name.
		/// </summary>
		Variable,
		/// <summary>
		/// Braces, commas.
		/// </summary>
		Delimiter,
		/// <summary>
		/// Registered signature.
		/// </summary>
		Signature
	}

	/// <summary>
	/// Represents a prepared sequence of precompiled items for compiling.
	/// </summary>
	public sealed class PreparedExpression
	{
		public List<PreparedExpressionItem> PreparedExpressionItems { get; private set; }

		public PreparedExpression(List<PreparedExpressionItem> preparedExpressionItems)
		{
			PreparedExpressionItems = preparedExpressionItems;
		}
	}

	/// <summary>
	/// Represents a part of parsed expression or decompiled expression.
	/// </summary>
	public struct PreparedExpressionItem//sealed class
	{
		public PreparedExpressionItemKind kind { get; private set; }
		public string variableOrConstant { get; private set; }
		public DelimiterKind delimiterKind { get; private set; }
		public string signature { get; private set; }

		public PreparedExpressionItem(PreparedExpressionItemKind kind, object value)
		{
			if (value == null) throw new ArgumentNullException("value");

			this.kind = kind;
			switch (kind)
			{
				case PreparedExpressionItemKind.Constant:
				case PreparedExpressionItemKind.Variable:
					variableOrConstant = (string) value;
					break;
				case PreparedExpressionItemKind.Delimiter:
					delimiterKind = (DelimiterKind) value;
					break;
				case PreparedExpressionItemKind.Signature:
					signature = (string) value;
					break;
				default:
					throw new InvalidOperationException("Unexpected item kind.");
			}
		}
	}
}
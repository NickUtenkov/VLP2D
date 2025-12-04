using System;
using System.Collections.Generic;

namespace MathSubstitutor
{
	/// <summary>
	/// Type of compiled expression item's content.
	/// </summary>
	public enum RPNItemKind
	{
		Constant = 1,
		Variable = 2,
		Operation = 3
	}

	/// <summary>
	/// Represents a compiled expression, which can be used for calculating.
	/// </summary>
	public sealed class RPNList
	{
		public List<RPNItem> RPNItems { get; private set; }

		public RPNList(List<RPNItem> _RPNItems)
		{
			RPNItems = _RPNItems;
		}
	}

	/// <summary>
	/// Item of compiled expression.
	/// </summary>
	public sealed class RPNItem
	{
		public RPNItemKind kind { get; private set; }
		public string variableOrConstant { get; private set; }
		public Operation operation { get; private set; }

		public RPNItem(RPNItemKind kind, object value)
		{
			if (value == null) throw new ArgumentNullException("value");
			this.kind = kind;
			switch (kind)
			{
				case RPNItemKind.Constant:
				case RPNItemKind.Variable:
					variableOrConstant = (string) value;
					break;
				case RPNItemKind.Operation:
					operation = (Operation)value;
					break;
				default:
					throw new InvalidOperationException("Unexpected kind.");
			}
		}
	}
}
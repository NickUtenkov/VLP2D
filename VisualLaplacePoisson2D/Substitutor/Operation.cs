using System;

namespace MathSubstitutor
{
	/// <summary>
	/// Type of operation.
	/// </summary>
	public enum OperationKind
	{
		Operator = 1,
		//Function = 2
	}

	/// <summary>
	/// Association direction of priority.
	/// </summary>
	public enum PriorityAssociation
	{
		/// <summary>
		/// For example, normal binary subtraction or multiplication.
		/// </summary>
		LeftAssociated = 1,
		/// <summary>
		/// For example, ternary ?: operator.
		/// </summary>
		RightAssociated = 2
	}

	/// <summary>
	/// Represents an operation with calculator associated.
	/// </summary>
	public struct Operation//sealed class
	{
		/// <summary>
		/// Operation name - unique string.
		/// </summary>
		public string name { get; private set; }

		/// <summary>
		/// Kind of operation.
		/// </summary>
		public OperationKind kind { get; private set; }

		/// <summary>
		/// Integer priority of operation.
		/// </summary>
		public int priority { get; private set; }

		/// <summary>
		/// Set of signature strings.
		/// </summary>
		public string[] signature { get; private set; }

		/// <summary>
		/// Count of operands.
		/// </summary>
		public int operandsCount { get; private set; }

		/// <summary>
		/// Calculator for this operation.
		/// </summary>
		public IOperationSubstitutor substitutor { get;}

		public Operation(string name, OperationKind kind, string[] signature, int operandsCount, IOperationSubstitutor substitutor, int priority)
		{
			if (name == null) throw new ArgumentNullException("name");
			if (name.Length == 0) throw new ArgumentException("Empty name.", "name");
			if (signature == null) throw new ArgumentNullException("signature");
			if (signature.Length == 0) throw new ArgumentException("Signature is empty.", "signature");
			if (substitutor == null) throw new ArgumentNullException("substitutor");
			if ((kind == OperationKind.Operator) && (operandsCount > 1) && (signature.Length != operandsCount - 1))
				throw new ArgumentException("Invalid array length.", "signature");
			/*if ((kind == OperationKind.Function) && (signature.Length != 1))
				throw new InvalidOperationException("Signature of function must contain one string item.");*/

			if (kind == OperationKind.Operator) this.priority = priority;
			this.kind = kind;
			this.name = name;
			this.substitutor = substitutor;
			this.operandsCount = operandsCount;
			this.signature = signature;
		}

		/*public Operation(string name, OperationKind kind, string[] signature, int operandsCount, IOperationSubstitutor substitutor)
		: this(name, kind, signature, operandsCount, calculator, 0)
		{
		}*/

		public override string ToString()
		{
			return String.Format("{0} {1}", kind, name);
		}
	}
}

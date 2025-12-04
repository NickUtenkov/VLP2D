using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace MathSubstitutor
{
	/// <summary>
	/// Supported operations and configuration manager.
	/// </summary>
	public sealed class OperationsRegistry
	{
		private readonly List<Operation> operationsList = new List<Operation>();
		private readonly Dictionary<int, PriorityAssociation> priorityAssociationsDictionary = new Dictionary<int, PriorityAssociation>();
		private readonly Dictionary<string, Operation> operationNamesDictionary = new Dictionary<string, Operation>();
		private readonly Dictionary<string, ICollection<Operation>> operationSignaturesDictionary = new Dictionary<string, ICollection<Operation>>();

		public bool IsPriorityDefined(int priority) => priorityAssociationsDictionary.ContainsKey(priority);

		public PriorityAssociation GetAssociationByPriority(int priority)
		{
			if (!IsPriorityDefined(priority)) throw new ArgumentException("Specified priority is not defined.", "priority");
			return priorityAssociationsDictionary[priority];
		}
		  
		public bool IsOperationDefined(string name)
		{
			if (name == null) throw new ArgumentNullException("name");
			if (name.Length == 0) throw new ArgumentException("String is empty.", "name");

			return operationNamesDictionary.ContainsKey(name);
		}

		public Operation GetOperationByName(string name)
		{
			if (!IsOperationDefined(name)) throw new ArgumentException("No operation defined with name specified.", "name");
			return operationNamesDictionary[name];
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Operation GetOperationByNameNoCheck(string name) => operationNamesDictionary[name];

		public bool IsSignatureDefined(string signature)
		{
			if (signature == null) throw new ArgumentNullException("signature");
			if (signature.Length == 0) throw new ArgumentException("String is empty.", "signature");

			return operationSignaturesDictionary.ContainsKey(signature);
		}

		public ICollection<Operation> GetOperationsUsingSignature(string signature)
		{
			if (!IsSignatureDefined(signature)) throw new ArgumentException("No operation uses signature specified.", "signature");

			return operationSignaturesDictionary[signature];
		}

		public int[] SignaturesLens { get; private set; }

		private void addOperator(string nameOperator, string[] signatures, int operandsCount, IOperationSubstitutor substitutor, int priority)
		{
			operationsList.Add(new Operation(nameOperator, OperationKind.Operator, signatures, operandsCount, substitutor, priority));
		}

		private void addOperators()
		{//below should be sorted by priority
			addOperator("powering", [ "^" ], 2, new OperatorPowering(), 1);
			addOperator("negation", ["-"], 1, new OperatorNegation(), 2);
			addOperator("positivation", ["+"], 1, new OperatorPositivation(), 2);
			addOperator("multiplication", ["*"], 2, new OperatorMultiplication(), 3);
			addOperator("division", ["/"], 2, new OperatorDivision(), 3);
			addOperator("addition", [ "+" ], 2, new OperatorAddition(), 4);
			addOperator("subtraction", [ "-" ], 2, new OperatorSubtraction(), 4);
			addOperator("conditional", [ "?", ":" ], 3, new OperatorConditional(), 5);
		}

		private void initialize()
		{
			priorityAssociationsDictionary.Add(1, PriorityAssociation.LeftAssociated);
			priorityAssociationsDictionary.Add(2, PriorityAssociation.RightAssociated);
			priorityAssociationsDictionary.Add(3, PriorityAssociation.LeftAssociated);
			priorityAssociationsDictionary.Add(4, PriorityAssociation.LeftAssociated);
			priorityAssociationsDictionary.Add(5, PriorityAssociation.RightAssociated);

			addOperators();
		}

		public OperationsRegistry()
		{
			initialize();
			// Storing signatures lengths has been met during processing
			List<int> lens = new List<int>();
			foreach (Operation operation in operationsList)
			{
				operationNamesDictionary.Add(operation.name, operation);

				foreach (string s in operation.signature)
				{
					if (!operationSignaturesDictionary.ContainsKey(s)) operationSignaturesDictionary.Add(s, new List<Operation>());
					operationSignaturesDictionary[s].Add(operation);
				}
				// Add signature lenght if not added already
				foreach (string s in operation.signature)
				{
					int len = s.Length;
					bool alreadySaved = false;
					foreach (int i in lens)
					{
						if (i == len)
						{
							alreadySaved = true;
							break;
						}
					}
					if (!alreadySaved) lens.Add(len);
				}
			}
			lens.Sort();
			SignaturesLens = new int[lens.Count];
			lens.CopyTo(SignaturesLens);
		}
	}
}

using Cloo;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class SimpleIterationAndRelaxationKernelBaseOCL
	{
		CommandQueueOCL commands;
		protected string strTypeName;
		protected string programSource;
		readonly string strDefines;
		readonly string defines =
@"
#define dimX	{0}
#define dimY	{1}
#define localDimX {2}
#define localDimY {3}
#define lastBlockSizeX	{4}
#define lastBlockSizeY	{5}
";

		public SimpleIterationAndRelaxationKernelBaseOCL(CommandQueueOCL commands, string strTypeName, int dimX, int dimY, int localDimX, int localDimY, int lastBlockSizeX, int lastBlockSizeY)
		{
			this.commands = commands;
			this.strTypeName = strTypeName;
			strDefines = string.Format(defines, dimX, dimY, localDimX, localDimY, lastBlockSizeX, lastBlockSizeY);
		}

		protected KernelOCL createKernel(string functionName, string argsIn, string strAction)
		{
			ProgramOCL program;
			string programName = UtilsCL.programName(functionName, strTypeName, commands.Device.VendorId);
			//program = UtilsCL.loadAndBuildProgram(programName, null, commands.Context, commands.Device);
			program = null;//should solve problem with defines

			if (program == null)
			{
				string args = string.Format(argsIn, strTypeName);
				string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;

				if (strTypeName == "DD128" || strTypeName == "QD256") strAction = ArithmeticReplacer.replaceArithmeticOperators(strAction);//not using replaceHPMacros because of one string only
				string strProgram = strDefines + strProgramHeader + formatSource(strAction);
				if (strTypeName == "DD128") strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
				if (strTypeName == "QD256") strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;

				program = UtilsCL.createProgram(strProgram, null, commands.Context, commands.Device);//"-cl-opt-disable" "-cl-std=CL3.0"
				//UtilsCL.saveProgram(programName, program.Binaries[0]);
			}
			return program.CreateKernel(functionName);
		}

		protected virtual string formatSource(string strAction)
		{
			return null;
		}
	}
}

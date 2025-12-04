
using Cloo;
using DD128Numeric;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class MultiGridSchemeOCL<T> : Iterative1DScheme<T>, IScheme<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		readonly BufferOCL<T>[] un;
		readonly BufferOCL<T>[] rhs;//right hand side
		readonly BufferOCL<T>[] res;//residual
		BufferOCL<T> fn;
		readonly T[] stepX, stepY;
		readonly SlidingIterationMultiGridSchemeOCL<T> smoother;
		readonly T eps;
		readonly int nLevels;
		readonly int[] dim0, dim1;
		T[] un0;
		long[][] workSize2DInternalPoints, workSize2DAllPoints, workSize1DInternalPointsX, workSize1DInternalPointsY;
		long[] workSizeEps = { 0, 0 }, workOffsetEps = { 0, 0 };
		BufferOCL<int> flagOCL;
		int[] flag = { 0 };
		bool un0Changed = false;
		T _8 = T.CreateTruncating(8);
		const int countSmoothingIterations = 3;//if 1 then no convergence in rectangle cases

		KernelOCL kernelResidual, kernelRestrictResidual, kernelFillArrayWithEdges, kernelInterpolate, kernelEpsExceeded;
		CommandQueueOCL commands;

		public MultiGridSchemeOCL(int cXSegments, int cYSegments, T stepXIn, T stepYIn, Func<T, T, T> fKsi, T eps, PlatformOCL platform, DeviceOCL device)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			commands = UtilsCL.createCommandQueue(platform, device, CommandQueueFlagsOCL.None);

			dimX = cXSegments + 1;
			dimY = cYSegments + 1;
			this.eps = eps;

			try
			{
				un0 = new T[dimX * dimY];

				if (fKsi != null)
				{
					GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => un0[i * dimY + j] = -fKsi(stepXIn * T.CreateTruncating(i), stepYIn * T.CreateTruncating(j)));
					fn = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, un0);
				}

				int nLevelsX = (int)Math.Log(cXSegments, 2);
				int nLevelsY = (int)Math.Log(cYSegments, 2);
				nLevels = Math.Min(nLevelsX, nLevelsY);

				dim0 = new int[nLevels];
				dim1 = new int[nLevels];
				un = new BufferOCL<T>[nLevels];
				rhs = new BufferOCL<T>[nLevels];
				res = new BufferOCL<T>[nLevels];
				stepX = new T[nLevels];
				stepY = new T[nLevels];
				smoother = new SlidingIterationMultiGridSchemeOCL<T>(commands);
				workSize2DInternalPoints = new long[nLevels][];
				workSize2DAllPoints = new long[nLevels][];
				workSize1DInternalPointsX = new long[nLevels][];
				workSize1DInternalPointsY = new long[nLevels][];

				int cSegsX = cXSegments;//is 2^n
				int cSegsY = cYSegments;//is 2^n
				T lngX = stepXIn * T.CreateTruncating(cSegsX);
				T lngY = stepYIn * T.CreateTruncating(cSegsY);
				for (int i = 0; i < nLevels; i++)
				{
					dim0[i] = cSegsX + 1;
					dim1[i] = cSegsY + 1;
					un[i] = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dim0[i] * dim1[i]);
					rhs[i] = (i == 0) && (fn != null) ? fn : new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dim0[i] * dim1[i]);
					res[i] = (i < nLevels - 1) ? new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dim0[i] * dim1[i]) : null;
					stepX[i] = lngX / T.CreateTruncating(cSegsX);
					stepY[i] = lngY / T.CreateTruncating(cSegsY);
					smoother.addKernelParameters(dim0[i], dim1[i], un[i], (i == 0) ? fn : rhs[i], stepX[i], stepY[i]);
					cSegsX /= 2;
					cSegsY /= 2;
					workSize2DInternalPoints[i] = new long[] { dim0[i] - 2, dim1[i] - 2 };
					if (i > 0) workSize2DAllPoints[i] = new long[] { dim0[i], dim1[i] };
					if (i > 0) workSize1DInternalPointsX[i] = new long[] { dim0[i] - 2 };
					if (i > 0) workSize1DInternalPointsY[i] = new long[] { dim1[i] - 2 };
				}
				if (fn == null) UtilsCL.initBuffer<T>(rhs[0], commands, T.Zero);
				flagOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadWrite, 1);
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			string funcNameResidual = "Residual";
			string funcNameRestrictResidual = "RestrictResidual";
			string funcNameFillArrayWithEdges = "FillArrayWithEdges";
			string funcNameInterpolate = "Interpolate";

			string programName = UtilsCL.programName("MultiGrid_", Utils.getTypeName<T>(), commands.Device.VendorId);
			ProgramOCL program = UtilsCL.loadAndBuildProgram(programName, null, commands.Context, commands.Device);

			if (program == null)
			{
				string strProgram = "";

				strProgram += createProgramResidual(funcNameResidual);
				strProgram += createProgramRestrictResidual(funcNameRestrictResidual);
				strProgram += createProgramFillArrayWithEdges(funcNameFillArrayWithEdges);
				strProgram += createProgramInterpolate(funcNameInterpolate);

				program = createProgram(strProgram);
				UtilsCL.saveProgram(programName, program.Binaries[0]);
			}

			ICollection<KernelOCL> kernels = program.CreateAllKernels();
			kernelResidual = kernels.First(x => x.FunctionName == funcNameResidual);
			kernelRestrictResidual = kernels.First(x => x.FunctionName == funcNameRestrictResidual);
			kernelFillArrayWithEdges = kernels.First(x => x.FunctionName == funcNameFillArrayWithEdges);
			kernelInterpolate = kernels.First(x => x.FunctionName == funcNameInterpolate);

			kernelEpsExceeded = UtilsCL.createProgramEpsExceeded(commands.Context, commands.Device, un[0], res[0], flagOCL, dimY, eps);
		}

		~MultiGridSchemeOCL()
		{
		}

		public T doIteration(int iter)
		{
			VCycle(0);
			un0Changed = true;

			return epsExceeded() ? eps + eps : T.Zero;
		}

		void VCycle(int startLevel)
		{
			for (int level = startLevel; level < nLevels - 1; level++)
			{
				smoother.doIterations(level, countSmoothingIterations);//fn(level=0),rhs[level](level>0),un[level] -> un[level]
				residual(level);//rhs[level],un[level] -> res[level]
				restrictResidual(level);//res[level + 0] -> rhs[level + 1]
				fillArrayWithEdges(level + 1, T.Zero);//un[level + 1] edges to val=0
			}
			smoother.doIterations(nLevels - 1, countSmoothingIterations);
			for (int level = nLevels - 2; level >= startLevel; level--)
			{
				interpolate(level);//un[level + 1] -> un[level + 0]
				if (level == 0) commands.CopyBuffer(un[0], res[0], null);//will be compare with(after this function)
				smoother.doIterations(level, countSmoothingIterations);
			}
		}

		string createProgramResidual(string functionName)
		{
			string args = string.Format("(global {0} *res, global {0} *rhs, global {0} *un, {0} stepX2, {0} stepY2, int dimY)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string programSourceResidual =
@"
{{
	int i = get_global_id(0) + 1;//indeces are 1-based, but workgroup indeces are 0-based
	int j = get_global_id(1) + 1;

	int idx = i * dimY + j;
	int idxim = idx - dimY;//im is 'i minus 1'
	int idxip = idx + dimY;//ip is 'i plus 1'
	int idxjm = idx - 1;//jm is 'j minus 1'
	int idxjp = idx + 1;//jp is 'j plus 1'

	//float centr = 2.0f * un[idx];
	res[idx] = HP(rhs[idx] - (un[idxim] + un[idxip] - 2.0 * un[idx]) / stepX2 - (un[idxjm] + un[idxjp] - 2.0 * un[idx]) / stepY2);
}}";
			return strProgramHeader + programSourceResidual;
		}

		void residual(int level)
		{
			kernelResidual.SetMemoryArgument(0, res[level]);
			kernelResidual.SetMemoryArgument(1, rhs[level]);
			kernelResidual.SetMemoryArgument(2, un[level]);
			kernelResidual.SetValueArgument(3, (stepX[level] * stepX[level]));
			kernelResidual.SetValueArgument(4, (stepY[level] * stepY[level]));
			kernelResidual.SetValueArgument(5, dim1[level]);

			commands.Execute(kernelResidual, null, workSize2DInternalPoints[level], null, null);
		}

		string createProgramRestrictResidual(string functionName)
		{
			string args = string.Format("(global {0} *rhs, global {0} *res, {0} oneDiv8, int rowSizeCoarseGrid, int rowSizeFineGrid)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string programSourceRestrictResidual =
@"
{{
	int i = get_global_id(0) + 1;//indeces are 1-based, but workgroup indeces are 0-based
	int j = get_global_id(1) + 1;

	int idx = i * rowSizeCoarseGrid + j;//coarse grid
	int idx1 = 2 * (i * rowSizeFineGrid + j);//fine grid

	int idxim = idx1 - rowSizeFineGrid;//im is 'i minus 1'
	int idxip = idx1 + rowSizeFineGrid;//ip is 'i plus 1'
	int idxjm = idx1 - 1;//jm is 'j minus 1'
	int idxjp = idx1 + 1;//jp is 'j plus 1'

	rhs[idx] = HP(oneDiv8 * (4.0 * res[idx1] + (res[idxip] + res[idxim] + res[idxjp] + res[idxjm])));
}}";
			return strProgramHeader + programSourceRestrictResidual;
		}

		void restrictResidual(int level)//Restriction
		{
			T oneDiv8 = T.One / _8;

			kernelRestrictResidual.SetMemoryArgument(0, rhs[level + 1]);
			kernelRestrictResidual.SetMemoryArgument(1, res[level + 0]);
			kernelRestrictResidual.SetValueArgument(2, oneDiv8);
			kernelRestrictResidual.SetValueArgument(3, dim1[level + 1]);//coarse
			kernelRestrictResidual.SetValueArgument(4, dim1[level + 0]);//fine

			commands.Execute(kernelRestrictResidual, null, workSize2DInternalPoints[level + 1], null, null);
		}

		string createProgramFillArrayWithEdges(string functionName)
		{
			string args = string.Format("(global {0} *un, {0} val)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string programSourceFillArrayWithEdges =
@"
{{
	int i = get_global_id(0);//indeces are 0-based, workgroup indeces are 0-based
	int j = get_global_id(1);
	int dim = get_global_size(1);

	int idx = i * dim + j;

	un[idx] = val;
}}";
			return strProgramHeader + programSourceFillArrayWithEdges;
		}

		void fillArrayWithEdges(int level, T val)//https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueFillBuffer.html
		{
			kernelFillArrayWithEdges.SetMemoryArgument(0, un[level]);
			kernelFillArrayWithEdges.SetValueArgument(1, val);

			commands.Execute(kernelFillArrayWithEdges, null, workSize2DAllPoints[level], null, null);
		}

		string createProgramInterpolate(string functionName)
		{
			string args = string.Format("(global {0} *uf, global {0} *uc, int rowSizeCoarseGrid, int rowSizeFineGrid)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCL.kernelPrefix + functionName + args;
			string programSourceInterpolate =
@"
{{
	int i = get_global_id(0);
	int j = get_global_id(1);

	int idxC1 = i * rowSizeCoarseGrid + j;
	int idxC2 = idxC1 + rowSizeCoarseGrid;

	int idx0 = 2 * (i * rowSizeFineGrid + j);//[i + 0, j + 0]
	int idx1 = idx0 + rowSizeFineGrid;//[i + 1, j + 0]

	uf[idx0 + 0] = HP(uf[idx0 + 0] + uc[idxC1]);//uc0[i, j]
	uf[idx1 + 0] = HP(uf[idx1 + 0] + 0.50 * (uc[idxC1] + uc[idxC2]));//0.50 * (uc0[i, j] + uc0[i + 1, j])
	uf[idx0 + 1] = HP(uf[idx0 + 1] + 0.50 * (uc[idxC1] + uc[idxC1 + 1]));//0.50 * (uc0[i, j] + uc0[i, j + 1])
	uf[idx1 + 1] = HP(uf[idx1 + 1] + 0.25 * ((uc[idxC1] + uc[idxC1 + 1]) + uc[idxC2] + uc[idxC2 + 1]));//0.25 * (uc0[i, j] + uc0[i, j + 1] + uc0[i + 1, j] + uc0[i + 1, j + 1])
}}";
			return strProgramHeader + programSourceInterpolate;
		}

		void interpolate(int level)
		{
			kernelInterpolate.SetMemoryArgument(0, un[level]);
			kernelInterpolate.SetMemoryArgument(1, un[level + 1]);
			kernelInterpolate.SetValueArgument(2, dim1[level + 1]);
			kernelInterpolate.SetValueArgument(3, dim1[level + 0]);

			long[] workSize2D = { workSize2DInternalPoints[level + 1][0] + 1 , workSize2DInternalPoints[level + 1][1] + 1 };
			commands.Execute(kernelInterpolate, null, workSize2D, null, null);
		}

		bool epsExceeded()
		{
			int cParts = 128;
			flag[0] = 0;
			commands.WriteToBuffer(flag, flagOCL, true, null);

			if (workSize2DInternalPoints[0][0] <= cParts)
			{
				commands.Execute(kernelEpsExceeded, null, workSize2DInternalPoints[0], null, null);
				commands.ReadFromBuffer(flagOCL, ref flag, true, null);
			}
			else
			{
				workOffsetEps[0] = 0;
				workSizeEps[1] = workSize2DInternalPoints[0][1];
				int wSize = (int)workSize2DInternalPoints[0][0];
				int partSize = wSize / cParts;
				workSizeEps[0] = partSize;
				int remainder = wSize % cParts;
				if (remainder > 0) cParts++;
				for (int i = 0; i < cParts; i++)
				{
					commands.Execute(kernelEpsExceeded, workOffsetEps, workSizeEps, null, null);
					commands.ReadFromBuffer(flagOCL, ref flag, true, null);
					if (flag[0] == 1) break;

					if ((i == cParts - 1) && (remainder != 0)) workSizeEps[0] = remainder;
					workOffsetEps[0] += partSize;
				}
			}

			return flag[0] == 1;
		}

		public override T[] getArray()
		{
			if (un0Changed)
			{
				un0Changed = false;

				commands.ReadFromBuffer(un[0], ref un0, true, null);
			}

			return un0;
		}

		public void initAfterBoundariesAndInitialIterationInited()
		{
			commands.WriteToBuffer(un0, un[0], true, null);
		}

		public int maxIterations() { return 0; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }

		public void cleanup()
		{
			UtilsCL.disposeQC(ref commands);

			UtilsCL.disposeKP(ref kernelRestrictResidual, false);
			UtilsCL.disposeKP(ref kernelFillArrayWithEdges, false);
			UtilsCL.disposeKP(ref kernelInterpolate, false);
			UtilsCL.disposeKP(ref kernelResidual);
			UtilsCL.disposeKP(ref kernelEpsExceeded);

			if (un != null) for (int i = 0; i <= un.GetUpperBound(0); i++) un[i].Dispose();
			if (res != null) for (int i = 0; i <= res.GetUpperBound(0); i++) if (res[i] != null) res[i].Dispose();
			if (rhs != null) for (int i = 0; i <= rhs.GetUpperBound(0); i++) if (rhs[i] != fn) rhs[i].Dispose();
			UtilsCL.disposeBuf(ref fn);
			UtilsCL.disposeBuf(ref flagOCL);
			un0 = null;
			smoother.cleanup();
		}

		ProgramOCL createProgram(string strProgram)
		{
			if (typeof(T) == typeof(float)) strProgram = HighPrecisionOCL.strSingleDefines + strProgram;
			else if (typeof(T) == typeof(double)) strProgram = HighPrecisionOCL.strDoubleDefines + strProgram;
			else strProgram = ArithmeticReplacer.replaceHPMacros(strProgram);
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;

			return UtilsCL.createProgram(strProgram, null, commands.Context, commands.Device);
		}
	}
}

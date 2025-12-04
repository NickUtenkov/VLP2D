using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	class SplittingSchemeCU<T> : ProgonkaSchemeCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>
	{
		public SplittingSchemeCU(int cXSegments, int cYSegments, T stepX, T stepY, T eps, Func<T, T, T> fKsi, int cudaDevice) :
			base(cXSegments, cYSegments, stepX, stepY, eps, false, fKsi, cudaDevice)
		{//S_VVCM p.258 (27) σ₁,σ₂
			T σ1 = T.CreateTruncating(0.5);
			T σ2 = T.CreateTruncating(0.5);
			T diagExtraX = stepX2 / (dt * σ1);
			T diagExtraY = stepY2 / (dt * σ2);

			calcAlpha(_2 + diagExtraX, _2 + diagExtraY);

			CUmodule? module;
			string moduleName = UtilsCU.moduleName("Splitting_", Utils.getTypeName<T>(), ctx.DeviceId);
			if (fnCU != null) moduleName += "_Fn";

			string functionNameX = "ProgonkaX";
			string functionNameY = "ProgonkaY";

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string strProgram = ProgonkaCU.strDefinesProgonkaXY;

				string constants = "static __device__ __constant__ " + Utils.getTypeName<T>() + " ";
				strProgram += constants + "srcCoefX, srcCoefY, operatorLxxCoef, operatorLyyCoef, fnCoefX, fnCoefY;\n";

				strProgram += SplittingProgramsCU.createProgramProgonkaX<T>(functionNameX, fnCU != null);
				strProgram += SplittingProgramsCU.createProgramProgonkaY<T>(functionNameY, fnCU != null);

				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}

			kernels[0] = new CudaKernel(functionNameX, (CUmodule)module);
			kernels[1] = new CudaKernel(functionNameY, (CUmodule)module);

			int upperX = dimX - 2;
			int upperY = dimY - 2;
			kernels[0].SetConstantVariable("dimX", dimX);
			kernels[0].SetConstantVariable("dimY", dimY);
			kernels[0].SetConstantVariable("upperX", upperX);
			kernels[0].SetConstantVariable("upperY", upperY);
			kernels[0].SetConstantVariable("srcCoefX", diagExtraX);
			kernels[0].SetConstantVariable("srcCoefY", diagExtraY);
			kernels[0].SetConstantVariable("operatorLxxCoef", (T.One - σ1) / σ1);
			kernels[0].SetConstantVariable("operatorLyyCoef", (T.One - σ2) / σ2);
			kernels[0].SetConstantVariable("fnCoefX", stepX2 / (σ1 * _2));
			kernels[0].SetConstantVariable("fnCoefY", stepY2 / (σ2 * _2));//

			List<object> argList0 = new List<object> { inputCU.DevicePointer, unmCU.DevicePointer, alphaXCU.DevicePointer};
			if (fnCU != null) argList0.Add(fnCU.DevicePointer);
			args[0] = argList0.ToArray();
			UtilsCU.set1DKernelDims(kernels[0], upperY);

			List<object> argList1 = new List<object> { unmCU.DevicePointer, outputCU.DevicePointer, alphaYCU.DevicePointer };
			if (fnCU != null) argList1.Add(fnCU.DevicePointer);
			args[1] = argList1.ToArray();
			UtilsCU.set1DKernelDims(kernels[1], upperX);
		}
	}
}

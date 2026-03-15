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
	class VarDirSchemeCU<T> : ProgonkaSchemeCU<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>
	{
		JordanSpeedup<T> jrd;

		public VarDirSchemeCU(RectangleData<T> rectData, T eps, Func<T, T, T> fKsi, bool isJordan, int cudaDevice) :
			base(rectData, eps, isJordan, fKsi, cudaDevice)
		{
			T ω = T.Zero;
			if (!isJordan)
			{
				ω = _2 / dt;
				calcAlpha(_2 + stepX2 * ω, _2 + stepY2 * ω);
			}
			else
			{
				jrd = new JordanSpeedup<T>(cXSegments, cYSegments, stepX2, stepY2, eps);

				calculateIterationAlpha = calcVariableDirectionsMethodAlpha;
			}

			CUmodule? module;
			string name = "VarDir_";
			if (fnCU != null) name += "_Fn";
			if (jrd != null) name += "_Jrd";
			string moduleName = UtilsCU.moduleName(name, Utils.getTypeName<T>(), ctx.DeviceId);

			string functionNameX = "ProgonkaX";
			string functionNameY = "ProgonkaY";

			module = UtilsCU.loadModule(moduleName, ctx);
#if !DEBUG//remove !
			if (module == null)
#endif
			{
				string strProgram = ProgonkaCU.strDefinesProgonkaXY;

				string constants = "static __device__ __constant__ " + Utils.getTypeName<T>() + " ";
				strProgram += constants + "stepX2, stepY2, oneDivY2, oneDivX2;\n";
				strProgram += VarDirProgramsCU.createProgramProgonkaX<T>(functionNameX, fnCU != null);
				strProgram += VarDirProgramsCU.createProgramProgonkaY<T>(functionNameY, fnCU != null);

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

			kernels[0].SetConstantVariable("stepX2", stepX2);//if (fnCU != null && jrd != null) 
			kernels[0].SetConstantVariable("oneDivY2", T.One / stepY2);

			kernels[1].SetConstantVariable("stepY2", stepY2);//if (fnCU != null && jrd != null) 
			kernels[1].SetConstantVariable("oneDivX2", T.One / stepX2);

			List<object> argList = new List<object> { inputCU.DevicePointer, unmCU.DevicePointer, alphaXCU.DevicePointer, ω };
			if (fnCU != null) argList.Add(fnCU.DevicePointer);
			args[0] = argList.ToArray();
			UtilsCU.set1DKernelDims(kernels[0], upperY);

			argList = new List<object> { unmCU.DevicePointer, outputCU.DevicePointer, alphaYCU.DevicePointer, ω };
			if (fnCU != null) argList.Add(fnCU.DevicePointer);
			args[1] = argList.ToArray();
			UtilsCU.set1DKernelDims(kernels[1], upperX);
		}

		public override int maxIterations() { return (jrd != null) ? jrd.maxIters : 0; }

		void calcVariableDirectionsMethodAlpha(int iter) => calcAlpha(_2 + stepX2 * jrd.w1(iter), _2 + stepY2 * jrd.w2(iter));

		protected override void setKernel0Arguments(int iter)
		{
			base.setKernel0Arguments(iter);
			if (jrd != null) args[0][3] = jrd.w1(iter);
		}

		protected override void setKernel1Arguments(int iter)
		{
			base.setKernel1Arguments(iter);
			if (jrd != null) args[1][3] = jrd.w2(iter);
		}

		public override IterationsKind iterationsKind()
		{
			return (jrd != null) ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}
	}
}

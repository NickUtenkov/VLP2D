using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using VLP2D.Common;

namespace VLP2D.Model
{
	class VarDirSchemeCU<T> : ProgonkaSchemeCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>
	{
		JordanSpeedup<T> jrd;

		public VarDirSchemeCU(int cXSegments, int cYSegments, T stepX, T stepY, T eps, Func<T, T, T> fKsi, bool isJordan, int cudaDevice) :
			base(cXSegments, cYSegments, stepX, stepY, eps, isJordan, fKsi, cudaDevice)
		{
			bool equalSteps = T.Abs(stepX - stepY) < T.Min(stepX, stepY) / T.CreateTruncating(100);

			T ω1 = T.Zero, ω2 = T.Zero;
			if (!isJordan)
			{
				ω1 = stepX2 * _2 / dt;
				ω2 = stepY2 * _2 / dt;

				calcAlpha(ω1 + _2, ω2 + _2);
			}
			else
			{
				jrd = new JordanSpeedup<T>(cXSegments, cYSegments, stepX2, stepY2, eps);

				calculateIterationAlpha = calcVariableDirectionsMethodAlpha;

				alphaXCU = new CudaDeviceVariable<T>(alphaX.GetLength(0));
				alphaYCU = new CudaDeviceVariable<T>(alphaY.GetLength(0));
			}

			CUmodule? module;
			string name = "VarDir_";
			if (fnCU != null) name += "_Fn";
			if (!equalSteps) name += "_DifSteps";
			if (jrd != null) name += "_Jrd";
			string moduleName = UtilsCU.moduleName(name, Utils.getTypeName<T>(), ctx.DeviceId);

			string functionNameX = "ProgonkaX";
			string functionNameY = "ProgonkaY";

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string strProgram = ProgonkaCU.strDefinesProgonkaXY;

				string constants = "static __device__ __constant__ " + Utils.getTypeName<T>() + " ";
				strProgram += constants + "stepX2, stepY2, stepX2DivY2, stepY2DivX2;\n";
				strProgram += VarDirProgramsCU.createProgramProgonkaX<T>(functionNameX, fnCU != null, equalSteps);
				strProgram += VarDirProgramsCU.createProgramProgonkaY<T>(functionNameY, fnCU != null, equalSteps);

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
			kernels[0].SetConstantVariable("stepX2DivY2", stepX2 / stepY2);//if (!equalSteps) 

			kernels[1].SetConstantVariable("stepY2", stepY2);//if (fnCU != null && jrd != null) 
			kernels[1].SetConstantVariable("stepY2DivX2", stepY2 / stepX2);//if (!equalSteps) 

			List<object> argList = new List<object> { inputCU.DevicePointer, unmCU.DevicePointer, alphaXCU.DevicePointer, ω1 };
			if (fnCU != null) argList.Add(fnCU.DevicePointer);
			args[0] = argList.ToArray();
			UtilsCU.set1DKernelDims(kernels[0], upperY);

			argList = new List<object> { unmCU.DevicePointer, outputCU.DevicePointer, alphaYCU.DevicePointer, ω2 };
			if (fnCU != null) argList.Add(fnCU.DevicePointer);
			args[1] = argList.ToArray();
			UtilsCU.set1DKernelDims(kernels[1], upperX);
		}

		public override int maxIterations() { return (jrd != null) ? jrd.maxIters : 0; }

		void calcVariableDirectionsMethodAlpha(int iter)
		{
			alphaX[0] = T.Zero;
			T w1kPlus2 = stepX2 * jrd.w1(iter) + _2;

			alphaY[0] = T.Zero;
			T w2kPlus2 = stepY2 * jrd.w2(iter) + _2;

			ParallelOptions optionsParallel = new ParallelOptions() { MaxDegreeOfParallelism = 2 };
			Parallel.For(0, 2, optionsParallel, j =>
			{
				if (j == 0) for (int i = 1; i < cXSegments; i++) alphaX[i] = T.One / (w1kPlus2 - alphaX[i - 1]);//[SNR] p.443, top;
				else for (int i = 1; i < cYSegments; i++) alphaY[i] = T.One / (w2kPlus2 - alphaY[i - 1]); ;
			});

			alphaXCU.CopyToDevice(alphaX);
			alphaYCU.CopyToDevice(alphaY);
		}

		protected override void setKernel0Arguments(int iter)
		{
			base.setKernel0Arguments(iter);
			if (jrd != null) args[0][3] = stepX2 * jrd.w1(iter);
		}

		protected override void setKernel1Arguments(int iter)
		{
			base.setKernel1Arguments(iter);
			if (jrd != null) args[1][3] = stepY2 * jrd.w2(iter);
		}

		public override IterationsKind iterationsKind()
		{
			return (jrd != null) ? IterationsKind.knownInAdvance : IterationsKind.unknown;
		}
	}
}

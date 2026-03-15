using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using VLP2D.Common;
using VLP2D.Model.CUDA;

namespace VLP2D.Model
{
	class EvolutionalFactorizationSchemeCU<T> : Iterative1DScheme<T>, IScheme<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>, IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>
	{
		T stepX2, stepY2, eps, hx2, hy2, xMax, yMax;
		CudaDeviceVariable<T> inputCU, outputCU, unmCU, fnCU, alphaXCU, alphaYCU;//fnCU unified memory
		T[] alphaX, alphaY;//beta is placed to result
		int cXSegments, cYSegments;

		CudaContext ctx;
		CudaKernel[] kernels;
		object[][] args;
		T[] un;
		bool unChanged = false;
		static T _05 = T.CreateTruncating(0.5), _2 = T.CreateTruncating(2), _4 = T.CreateTruncating(4);
		T[] lambda_max, lambda_min, τ;
		T conditioning;

		public EvolutionalFactorizationSchemeCU(RectangleData<T> rectData, T epsIn, Func<T, T, T> fKsi, int cudaDevice) :
			base(rectData.xMin, rectData.yMin, rectData.stepX, rectData.stepY)
		{
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			cXSegments = rectData.cXSegments;
			cYSegments = rectData.cYSegments;
			xMax = rectData.xMax;
			yMax = rectData.yMax;

			dimX = cXSegments + 1;
			dimY = cYSegments + 1;

			un = new T[dimX * dimY];

			alphaX = new T[cXSegments];
			alphaY = new T[cYSegments];

			eps = epsIn;

			ctx = new CudaContext(cudaDevice);
			try
			{
				inputCU = new CudaDeviceVariable<T>(dimX * dimY);
				outputCU = new CudaDeviceVariable<T>(dimX * dimY);
				unmCU = new CudaDeviceVariable<T>(dimX * dimY);

				alphaXCU = new CudaDeviceVariable<T>(alphaX.GetLength(0));
				alphaYCU = new CudaDeviceVariable<T>(alphaY.GetLength(0));
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
			kernels = new CudaKernel[2];
			args = new object[2][];

			if (fKsi != null)
			{
				T[] fnFloat = new T[dimX * dimY];//exterior points are not used; can't iterate on fnCU, throws exceptions
				GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => fnFloat[i * dimY + j] = fKsi(xMin + stepX * T.CreateTruncating(i), yMin + stepY * T.CreateTruncating(j)));
				fnCU = fnFloat;
				fnFloat = null;
			}

			T ratioOfSquaresOfSteps = stepX2 / stepY2;
			hx2 = stepX2 / _4;//== (stepX/2)^2
			hy2 = stepY2 / _4;//== (stepY/2)^2

			τ = calculateTau();

			CUmodule? module;
			string name = "EvolFactor_";
			if (fnCU != null) name += "_Fn";
			string moduleName = UtilsCU.moduleName(name, Utils.getTypeName<T>(), ctx.DeviceId);

			string functionNameX = "ProgonkaX";
			string functionNameY = "ProgonkaY";

			module = UtilsCU.loadModule(moduleName, ctx);
#if !DEBUG//remove !?
			if (module == null)
#endif
			{
				string strProgram = EvolutionalFactorizationProgramsCU.strDefinesProgonkaXY;

				string constants = "static __device__ __constant__ " + Utils.getTypeName<T>() + " ";
				strProgram += constants + "stepX2, hy2, ratio;\n";
				strProgram += EvolutionalFactorizationProgramsCU.createProgramProgonkaX<T>(functionNameX, fnCU != null);
				strProgram += EvolutionalFactorizationProgramsCU.createProgramProgonkaY<T>(functionNameY);

				if (typeof(T) == typeof(float)) strProgram = HighPrecisionCU.strSingleDefines + strProgram;
				if (typeof(T) == typeof(double)) strProgram = HighPrecisionCU.strDoubleDefines + strProgram;
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

			kernels[0].SetConstantVariable("stepX2", stepX2);
			kernels[0].SetConstantVariable("ratio", ratioOfSquaresOfSteps);

			kernels[1].SetConstantVariable("hy2", hy2);

			List<object> argList = new List<object> { inputCU.DevicePointer, unmCU.DevicePointer, alphaXCU.DevicePointer, T.Zero };
			if (fnCU != null) argList.Add(fnCU.DevicePointer);
			args[0] = argList.ToArray();
			UtilsCU.set1DKernelDims(kernels[0], upperY);

			argList = new List<object> { unmCU.DevicePointer, outputCU.DevicePointer, inputCU.DevicePointer, alphaYCU.DevicePointer, T.Zero, T.Zero };
			args[1] = argList.ToArray();
			UtilsCU.set1DKernelDims(kernels[1], upperX);
		}

		public T doIteration(int iter)
		{
			unChanged = true;
			evolutionalFactorization(τ[iter]);

			UtilsSwap.swap(ref inputCU, ref outputCU);

			return T.One;
		}

		void evolutionalFactorization(T τ)
		{
			T twoDivTau = _2 / τ;

			calcAlpha(_2 + twoDivTau * hx2, _2 + twoDivTau * hy2);

			setKernel0Arguments(twoDivTau);
			kernels[0].Run(args[0]);

			setKernel1Arguments(twoDivTau, τ);
			kernels[1].Run(args[1]);
		}

		void calcAlpha(T bx, T by)
		{
			Action<T, T[], int> calc = (diag, alpha, bound) =>
			{
				alpha[0] = T.Zero;
				for (int i = 1; i <= bound; i++) alpha[i] = T.One / (diag - alpha[i - 1]);
			};

			Action[] actions = [() => calc(bx, alphaX, cXSegments - 1), () => calc(by, alphaY, cYSegments - 1)];
			Parallel.For(0, 2, GridIterator.optionsParallel, j => actions[j].Invoke());

			alphaXCU.CopyToDevice(alphaX);
			alphaYCU.CopyToDevice(alphaY);
		}

		void setKernel0Arguments(T twoDivTau)
		{ 
			args[0][0] = inputCU.DevicePointer;
			args[0][3] = twoDivTau;
		}
		void setKernel1Arguments(T twoDivTau, T τ)
		{
			args[1][1] = outputCU.DevicePointer;
			args[1][2] = inputCU.DevicePointer;
			args[1][4] = twoDivTau;
			args[1][5] = τ;
		}

		public override T[] getArray()
		{
			if (unChanged)
			{
				unChanged = false;

				inputCU.CopyToHost(un);
			}

			return un;
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
			inputCU.CopyToDevice(un);
			outputCU.CopyToDevice(inputCU);
		}

		public virtual int maxIterations() { return τ.Length; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }
		public override IterationsKind iterationsKind() => IterationsKind.knownInAdvance;

		public virtual void cleanup()
		{
			UtilsCU.disposeBuf(ref inputCU);
			UtilsCU.disposeBuf(ref outputCU);
			UtilsCU.disposeBuf(ref unmCU);
			UtilsCU.disposeBuf(ref fnCU);

			if (kernels != null) ctx?.UnloadModule(kernels[0].CUModule);
			ctx?.Dispose();
			ctx = null;
		}

		T[] calculateTau()
		{
			(lambda_max, lambda_min, conditioning) = spectrumEstimates();
			T epsilon_background = T.CreateTruncating(UtilsEps.epsilon<T>()) * conditioning;//can use UtilsEps.epsilonBackground<T>()
			T eps_grid_sys = eps;
			T epsilon = T.Max(epsilon_background, eps_grid_sys);
			int[] sAll = numberOfSteps(double.CreateTruncating(conditioning), double.CreateTruncating(epsilon));
			List<T> listτ = new List<T>();
			for (int i = 0; i < sAll.Length; i++)
			{
				int S = sAll[i];
				T[] tau_lt = logarithmicSet(lambda_max, lambda_min, S);
				if (i == 0) listτ.AddRange(tau_lt);
				else for (int s = 1; s <= S / 2; s++) listτ.Add(tau_lt[2 * s - 1]);
			}

			return listτ.ToArray();
		}

		(T[], T[], T) spectrumEstimates()
		{//Constructs estimates for spectrum boundaries of the grid system
			T est_x = _4 / hx2;
			T est_y = _4 / hy2;
			T[] lambda_max = [est_x, est_y];
			T lx = xMax - xMin;
			T ly = yMax - yMin;
			T pilx = T.Pi / lx;
			T pily = T.Pi / ly;
			T[] lambda_min = [pilx * pilx, pily * pily];
			T conditioning = (lambda_max[0] + lambda_max[1]) / (lambda_min[0] + lambda_min[1]);
			return (lambda_max, lambda_min, conditioning);
		}

		int[] numberOfSteps(double relation, double epsilon)
		{// Calculates the numbers of steps in nested logarithmic grids via a priori convergence estimate
			int count = 10;
			double[] S_temp = new double[count];
			int I = 0;
			S_temp[0] = double.Ceiling(-(4 / (Math.PI * Math.PI + 2 * Math.PI)) * Math.Log(relation) * Math.Log(epsilon));
			while (S_temp[I] >= 2)
			{
				S_temp[I + 1] = S_temp[I] / 2;
				I = I + 1;
				if (I == count) break;
			}

			int[] sAll = new int[I];
			double ceil = double.Ceiling(S_temp[I]);
			for (int m = 0; m < I; m++) sAll[m] = (int)(ceil * Math.Pow(2, m));

			return sAll;
		}

		T[] logarithmicSet(T[] lambda_max, T[] lambda_min, int S)
		{// Constructs linear-trigonometric set in logarithmic scale with given number of steps
			T τMin = _2 / (lambda_max[0] + lambda_max[1]);
			T τMax = _2 / (lambda_min[0] + lambda_min[1]);
			T center = _05 * T.Log(τMin * τMax);
			T width = _05 * T.Log(τMax / τMin);
			T piPlus2 = T.Pi + _2;
			T _2Div = _2 / piPlus2;
			T piDiv = T.Pi / piPlus2;

			T[] τ = new T[S + 1];
			T _S = T.CreateTruncating(S);
			for (int s = 0; s <= S; s++)
			{
				T θ = T.CreateTruncating(s) / _S;
				T lt = _2Div * T.Cos(θ * T.Pi - T.Pi) + piDiv * (_2 * θ - T.One);
				τ[s] = T.Pow(T.E, center + width * lt);
			}

			return τ;
		}
	}
}

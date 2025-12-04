
using Cloo;
using DD128Numeric;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using VLP2D.Properties;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	struct MethodsParams
	{
		public bool isJordan, isChebysh, isBiconjugateStabilized, isVarSepProgonka, isCompareAnalytic;
		public int paramL;
		public VarSepMethodsEnum methodVarSep;
		public CRMethodsEnum methodCR;
		public PlatformOCL platform;
		public DeviceOCL device;
		public int cudaDevice;
	}

	class VLPRectangleModel<T> : IVLPRectangleModel where T :
		unmanaged, INumber<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>,
		IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>
	{
		T stepX, stepY;
		int cXSegments, cYSegments;
		int elapsedIters, maxIters = -7;
		IterationsKind iterationsKind;
		T eps;
		//above is the same(except name) as VLPRectangleParams(but not using them in order not use pointer to get values)
		T fMinDiff = T.MaxValue, fMaxDiff = T.MinValue;
		double initTime, elapsedTime;
		string strElapsedInfo;
		IScheme<T> scheme;

		T[][] unDiff;
		int degreeOfParallelism = Environment.ProcessorCount;

		event progressDelegateLPRectangle progressEvent;
		event progressHeaderDelegateLPRectangle progressHeaderEvent;
		event completedDelegateLPRectangle completedEvent;
		bool bCancelAllIterations = false;
		BackgroundWorker worker;
		List<BitmapSource> lstBitmap, lstBitmapDiff;
		Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap;
		Func<MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmapDiff;
		bool visualize;
		int stepHeatMap;
		ParallelOptions optionsParallel;
		PlatformAndSchemeIndex platformScheme;
		InterpolationEnumVLPRectangle indexInterpolation;
		MethodsParams methodParams;
		StepsRecalculator<T> stepsRecalculator;

		CompiledFunctions<T> compiledFuncs;
		Func<T, T> pFuncLeft, pFuncRight, pFuncTop, pFuncBottom;//for skip internal null check in CompiledFunctions
		Func<T, T, T> pFuncKsi, pFuncBoundary, pFuncAnalytic;//for skip internal null check in CompiledFunctions

		public VLPRectangleModel()
		{
			optionsParallel = new ParallelOptions();
		}

		public void recalculateSteps(VLPRectangleParams pParams, PlatformAndSchemeIndex platformScheme, bool isVarSepProgonkaGPU, VarSepMethodsEnum varSepMethodCPU)
		{
			stepsRecalculator = new StepsRecalculator<T>();
			stepsRecalculator.recalculateSteps(pParams.xMax, pParams.cXSegmentsOriginal, pParams.yMax, pParams.cYSegmentsOriginal, platformScheme, isVarSepProgonkaGPU, varSepMethodCPU);

			pParams.stepX = double.CreateTruncating(stepsRecalculator.stepX);
			pParams.stepY = double.CreateTruncating(stepsRecalculator.stepY);
			pParams.cXSegments = stepsRecalculator.cXSegments;
			pParams.cYSegments = stepsRecalculator.cYSegments;
		}

		public void setMaxIterations(int maxIterations)
		{
			maxIters = maxIterations;
		}

		public void compileFunctions(VLPRectangleParams pParams)
		{
			compiledFuncs = new CompiledFunctions<T>(pParams.funcLeft, pParams.funcRight, pParams.funcTop, pParams.funcBottom, pParams.fKsi, pParams.funcBoundary, pParams.funcAnalytic, true);
			pFuncLeft = compiledFuncs.pFuncLeft;
			pFuncRight = compiledFuncs.pFuncRight;
			pFuncTop = compiledFuncs.pFuncTop;
			pFuncBottom = compiledFuncs.pFuncBottom;
			pFuncKsi = compiledFuncs.pFuncKsi;
			pFuncBoundary = compiledFuncs.pFuncBoundary;
			pFuncAnalytic = compiledFuncs.pFuncAnalytic;
		}

		public void prepareCalculation(VLPRectangleParams pParams, InterpolationEnumVLPRectangle idxInterpolation, PlatformAndSchemeIndex platformScheme, List<BitmapSource> lstBitmap, List<BitmapSource> lstBitmapDiff, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Func<MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmapDiff)
		{
			this.platformScheme = platformScheme;
			indexInterpolation = idxInterpolation;

			stepX = stepsRecalculator.stepX;
			stepY = stepsRecalculator.stepY;
			cXSegments = stepsRecalculator.cXSegments;
			cYSegments = stepsRecalculator.cYSegments;

			eps = getEpsilon(pParams.dictEps);
			this.lstBitmap = lstBitmap;
			this.fCreateBitmap = fCreateBitmap;

			float newStepX = 0, newStepY = 0;
			double[,] uu = createInitialBitmap(pParams, idxInterpolation, ref newStepX, ref newStepY);

			if (methodParams.isCompareAnalytic)
			{
				CompiledFunctions<double> compiledFuncsFloat = new CompiledFunctions<double>(null, null, null, null, null, null, pParams.funcAnalytic);

				if (compiledFuncsFloat.pFuncAnalytic != null)
				{
					int dim1 = uu.GetUpperBound(0) + 1;
					int dim2 = uu.GetUpperBound(1) + 1;
					double[][] unDiffLocal = new double[dim1][];
					for (int i = 0; i < dim1; i++) unDiffLocal[i] = new double[dim2];
					if (unDiffLocal != null)
					{
						double fMinDiffLocal = double.MaxValue, fMaxDiffLocal = double.MinValue;
						this.lstBitmapDiff = lstBitmapDiff;
						this.fCreateBitmapDiff = fCreateBitmapDiff;
						Adapter2D<double> adapter = new Adapter2D<double>(dim1, dim2, (i, j) => uu[i, j]);
						UtilsDiff.calculateDifference<double>(adapter, unDiffLocal, newStepX, newStepY, compiledFuncsFloat.pFuncAnalytic, ref fMinDiffLocal, ref fMaxDiffLocal, null, null);
						BitmapSource bmsDiff = fCreateBitmapDiff(new MinMaxF(fMinDiffLocal, fMaxDiffLocal), new Adapter2D<float>(dim1, dim2, (i, j) => (float)unDiffLocal[i][j]));
						if (bmsDiff != null) lstBitmapDiff.Add(bmsDiff);
					}
				}
			}
		}

		public void setMethodParams(MethodsParams mParams)
		{
			methodParams = mParams;
		}

		public int getMaxIterations(PlatformAndSchemeIndex platformScheme, VLPRectangleParams pParams, InterpolationEnumVLPRectangle idxInterpolation)
		{
			int rc = 0;
			if (platformScheme.platrofm == PlatformEnum.CPU)
			{
				string enumName = Enum.GetName(typeof(SchemeCPUEnum), platformScheme.idxScheme);
				rc = getIterations(pParams.maxCPUIters, enumName, (int)idxInterpolation);
			}
			else if (platformScheme.platrofm == PlatformEnum.OCL)
			{
				string enumName = Enum.GetName(typeof(SchemeOCLEnum), platformScheme.idxScheme);
				rc = getIterations(pParams.maxOCLIters, enumName, (int)idxInterpolation);
			}
			else if (platformScheme.platrofm == PlatformEnum.CUDA)
			{
				string enumName = Enum.GetName(typeof(SchemeCUDAEnum), platformScheme.idxScheme);
				rc = getIterations(pParams.maxCUDAIters, enumName, (int)idxInterpolation);
			}

			return rc;
		}

		T getEpsilon(Dictionary<string, double> dictEps)
		{
			string key = "epsilon";
			string typeName = typeof(T).Name;
			if (dictEps.ContainsKey(key + typeName)) return T.CreateTruncating(dictEps[key + typeName]);
			if (dictEps.ContainsKey(key)) return T.CreateTruncating(dictEps[key]);
			if (typeof(T) == typeof(float)) return T.CreateTruncating(1E-4);
			if (typeof(T) == typeof(double)) return T.CreateTruncating(1E-13);
			if (typeof(T) == typeof(DD128)) return T.CreateTruncating(1E-28);
			if (typeof(T) == typeof(QD256)) return T.CreateTruncating(1E-58);
			return T.Zero;
		}

		int getIterations(Dictionary<string, int[]> dictIters, string keyIters, int idxInterpolation)
		{
			string typeName = typeof(T).Name;
			if (keyIters != null)
			{
				if (dictIters.ContainsKey(keyIters + typeName)) return dictIters[keyIters + typeName][idxInterpolation];
				if (dictIters.ContainsKey(keyIters)) return dictIters[keyIters][idxInterpolation];
			}
			return 5000;
		}

		double[,] createInitialBitmap(VLPRectangleParams pParams, InterpolationEnumVLPRectangle idxInterpolation,ref float newStepX, ref float newStepY)
		{
			CompiledFunctions<double> compiledFuncsFloat = new CompiledFunctions<double>(pParams.funcLeft, pParams.funcRight, pParams.funcTop, pParams.funcBottom, null, pParams.funcBoundary, null);
			Func<double, double> pFuncLeftFloat, pFuncRightFloat, pFuncTopFloat, pFuncBottomFloat;
			Func<double, double, double> pFuncBoundaryFloat;
			pFuncLeftFloat = compiledFuncsFloat.pFuncLeft;
			pFuncRightFloat = compiledFuncsFloat.pFuncRight;
			pFuncTopFloat = compiledFuncsFloat.pFuncTop;
			pFuncBottomFloat = compiledFuncsFloat.pFuncBottom;
			pFuncBoundaryFloat = compiledFuncsFloat.pFuncBoundary;

			double ratio = (double)cYSegments / cXSegments;
			int cXSegsBmp = (cXSegments > 200) ? 200 : cXSegments;
			int cYSegsBmp = (int)(ratio * cXSegsBmp);
			double[,] uu = new double[cXSegsBmp + 1, cYSegsBmp + 1];
			newStepX = float.CreateTruncating(stepX) * cXSegments / cXSegsBmp;
			newStepY = float.CreateTruncating(stepY) * cYSegments / cYSegsBmp;

			double bMin = double.MaxValue;
			double bMax = double.MinValue;
			UtilsBorders.initTopBottomBorders<double>(uu, newStepX, newStepY, pFuncBottomFloat, pFuncTopFloat, pFuncBoundaryFloat, ref bMin, ref bMax);
			UtilsBorders.initLeftRightBorders<double>(uu, newStepX, newStepY, pFuncLeftFloat, pFuncRightFloat, pFuncBoundaryFloat, ref bMin, ref bMax);

			if (idxInterpolation == InterpolationEnumVLPRectangle.Mean) UtilsII.initInitialIterationMean<double>(uu, (bMin + bMax) / 2.0f);
			else if (idxInterpolation == InterpolationEnumVLPRectangle.ArithmeticMean) UtilsII.initInitialIterationArithmeticMean(uu);
			else if (idxInterpolation == InterpolationEnumVLPRectangle.Linear) UtilsII.initInitialIterationLinearInterpolation(uu);
			else if (idxInterpolation == InterpolationEnumVLPRectangle.WeightLinear) UtilsII.initInitialIterationWeightLinearInterpolation(uu);

			BitmapSource bms = fCreateBitmap(false, new MinMaxF(bMin, bMax), new Adapter2D<float>(cXSegsBmp + 1, cYSegsBmp + 1, (i, j) => (float)uu[i, j]));
			if (bms != null) lstBitmap.Add(bms);
			return uu;
		}

		void createScheme(PlatformAndSchemeIndex platformScheme, MethodsParams mParams)
		{
			try
			{
				if (platformScheme.platrofm == PlatformEnum.CPU)
				{
					SchemeCPUEnum idxScheme = (SchemeCPUEnum)platformScheme.idxScheme;
					if (idxScheme == SchemeCPUEnum.SimpleIteration)
					{
						scheme = new SimpleIterationScheme<T>(cXSegments, cYSegments, stepX, stepY, mParams.isChebysh, eps, pFuncKsi);
						if (mParams.isChebysh) maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCPUEnum.SlidingIteration)
					{
						scheme = new RelaxationScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, true, false, eps);
					}
					else if (idxScheme == SchemeCPUEnum.SOR)
					{
						scheme = new RelaxationScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, false, false, eps);
					}
					else if (idxScheme == SchemeCPUEnum.Splitting)
					{
						scheme = new SplittingScheme<T>(cXSegments, cYSegments, stepX, stepY, eps, pFuncKsi, optionsParallel);
					}
					else if (idxScheme == SchemeCPUEnum.VarDir)
					{
						scheme = new VarDirScheme<T>(cXSegments, cYSegments, stepX, stepY, eps, pFuncKsi, optionsParallel, mParams.isJordan);
						if (mParams.isJordan) maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCPUEnum.PTM)
					{
						scheme = new PTMScheme<T>(cXSegments, cYSegments, stepX, stepY, mParams.isChebysh, eps, pFuncKsi);
						if (mParams.isChebysh) maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCPUEnum.GradientDescent)
					{
						scheme = new GradientDescentScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, eps);
					}
					else if (idxScheme == SchemeCPUEnum.MinimumResidual)
					{
						scheme = new MinimumResidualScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, eps);
					}
					else if (idxScheme == SchemeCPUEnum.ConjugateGradient)
					{
						scheme = new ConjugateGradientScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, eps);
					}
					else if (idxScheme == SchemeCPUEnum.BiconjugateGradient)
					{
						if (mParams.isBiconjugateStabilized) scheme = new BiconjugateStabilizedScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi);
						else scheme = new BiconjugateScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, eps);
					}
					else if (idxScheme == SchemeCPUEnum.Chebishev3Layers)
					{
						scheme = new ChebyshevIterationScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, eps);
					}
					else if (idxScheme == SchemeCPUEnum.MultiGrid)
					{
						scheme = new MultiGridScheme<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, eps);
					}
					else if (idxScheme == SchemeCPUEnum.CompleteReduction)
					{
						int cCores = optionsParallel.MaxDegreeOfParallelism;
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						if (mParams.methodCR == CRMethodsEnum.SN)
						{
							scheme = new CyclicReductionSamarskiiNikolaevScheme<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, lst, fCreateBitmap, reportProgress);
							//scheme = new CyclicReductionSamarskiiNikolaevScheme_BothPrecision<T>(cXSegments, cYSegments, stepX, stepY, true, cCores, pFuncKsi, lst, fCreateBitmap, reportProgress);
						}
						else
						{
							scheme = new CyclicReductionBunemanScheme<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, lst, fCreateBitmap, reportProgress);
						}
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCPUEnum.VariablesSeparation)
					{
						int cCores = optionsParallel.MaxDegreeOfParallelism;
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						if (mParams.methodVarSep != VarSepMethodsEnum.FFT)
						{
							scheme = new VariablesSeparationSchemeProgonka<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, mParams.methodVarSep, lst, fCreateBitmap, reportProgress);
						}
						else
						{
							scheme = new VariablesSeparationSchemeNoProgonka<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, lst, fCreateBitmap, reportProgress);
						}
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCPUEnum.IncompleteReduction)
					{
						int cCores = optionsParallel.MaxDegreeOfParallelism;
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new FACRScheme<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, mParams.paramL, lst, fCreateBitmap, reportProgress);
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCPUEnum.MatrixProgonka)
					{
						int cCores = optionsParallel.MaxDegreeOfParallelism;
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new MatrixProgonkaScheme<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, lst, fCreateBitmap, reportProgress);
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCPUEnum.Marching)
					{
						int cCores = optionsParallel.MaxDegreeOfParallelism;
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new MarchingScheme<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, mParams.paramL, lst, fCreateBitmap, reportProgress);
						maxIters = scheme.maxIterations();
					}
				}
				else if (platformScheme.platrofm == PlatformEnum.OCL)
				{
					SchemeOCLEnum idxScheme = (SchemeOCLEnum)platformScheme.idxScheme;
					if (idxScheme == SchemeOCLEnum.SimpleIterationOCL)
					{
						scheme = new SimpleIterationSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, mParams.isChebysh, eps, pFuncKsi, mParams.platform, mParams.device);
						if (mParams.isChebysh) maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeOCLEnum.SlidingIterationOCL)
					{
						scheme = new RelaxationSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, true, false, eps, mParams.platform, mParams.device);
					}
					else if (idxScheme == SchemeOCLEnum.SOROCL)
					{
						scheme = new RelaxationSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, false, false, eps, mParams.platform, mParams.device);
					}
					else if (idxScheme == SchemeOCLEnum.SplittingOCL)
					{
						scheme = new SplittingSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, eps, pFuncKsi, mParams.platform, mParams.device);
					}
					else if (idxScheme == SchemeOCLEnum.VarDirOCL)
					{
						scheme = new VarDirSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, eps, pFuncKsi, mParams.isJordan, mParams.platform, mParams.device);
						if (mParams.isJordan) maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeOCLEnum.MultiGridOCL)
					{
						scheme = new MultiGridSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, eps, mParams.platform, mParams.device);
					}
					else if (idxScheme == SchemeOCLEnum.CompleteReductionOCL)
					{
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new CyclicReductionSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, lst, fCreateBitmap, reportProgress, mParams.platform, mParams.device);
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeOCLEnum.VariablesSeparationOCL)
					{
						if (mParams.isVarSepProgonka)
						{
							scheme = new VariablesSeparationSchemeProgonkaOCL<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, optionsParallel, mParams.platform, mParams.device, reportProgress);
						}
						else
						{
							scheme = new VariablesSeparationSchemeNoProgonkaOCL<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, optionsParallel, mParams.platform, mParams.device, reportProgress);
						}
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeOCLEnum.IncompleteReductionOCL)
					{
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new FACRSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, mParams.paramL, optionsParallel, lst, fCreateBitmap, mParams.platform, mParams.device, reportProgress);
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeOCLEnum.MarchingOCL)
					{
						int cCores = optionsParallel.MaxDegreeOfParallelism;
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new MarchingSchemeOCL<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, mParams.paramL, lst, fCreateBitmap, mParams.platform, mParams.device, reportProgress);
						maxIters = scheme.maxIterations();
					}
				}
				else if (platformScheme.platrofm == PlatformEnum.CUDA)
				{
					SchemeCUDAEnum idxScheme = (SchemeCUDAEnum)platformScheme.idxScheme;
					if (idxScheme == SchemeCUDAEnum.SimpleIterationCUDA)
					{
						scheme = new SimpleIterationSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, mParams.isChebysh, eps, pFuncKsi, mParams.cudaDevice);
						if (mParams.isChebysh) maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCUDAEnum.SlidingIterationCUDA)
					{
						scheme = new RelaxationSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, true, eps, mParams.cudaDevice);
					}
					else if (idxScheme == SchemeCUDAEnum.SORCUDA)
					{
						scheme = new RelaxationSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, false, eps, mParams.cudaDevice);
					}
					else if (idxScheme == SchemeCUDAEnum.SplittingCUDA)
					{
						scheme = new SplittingSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, eps, pFuncKsi, mParams.cudaDevice);
					}
					else if (idxScheme == SchemeCUDAEnum.VarDirCUDA)
					{
						scheme = new VarDirSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, eps, pFuncKsi, mParams.isJordan, mParams.cudaDevice);
						if (mParams.isJordan) maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCUDAEnum.MultiGridCUDA)
					{
						scheme = new MultiGridSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, eps, mParams.cudaDevice);
					}
					else if (idxScheme == SchemeCUDAEnum.CompleteReductionCUDA)
					{
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new CyclicReductionSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, lst, fCreateBitmap, reportProgress, mParams.cudaDevice);
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCUDAEnum.VariablesSeparationCUDA)
					{
						if (mParams.isVarSepProgonka) scheme = new VariablesSeparationSchemeProgonkaCU<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, optionsParallel, reportProgress, mParams.cudaDevice);
						else scheme = new VariablesSeparationSchemeNoProgonkaCU<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, optionsParallel, reportProgress, mParams.cudaDevice);
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCUDAEnum.FACRCUDA)
					{
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new FACRSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, pFuncKsi, mParams.paramL, lst, fCreateBitmap, optionsParallel, reportProgress, mParams.cudaDevice);
						maxIters = scheme.maxIterations();
					}
					else if (idxScheme == SchemeCUDAEnum.MarchingCUDA)
					{
						int cCores = optionsParallel.MaxDegreeOfParallelism;
						List<BitmapSource> lst = visualize ? lstBitmap : null;
						scheme = new MarchingSchemeCU<T>(cXSegments, cYSegments, stepX, stepY, cCores, pFuncKsi, mParams.paramL, lst, fCreateBitmap, reportProgress, mParams.cudaDevice);
						maxIters = scheme.maxIterations();
					}
				}
			}
			catch (Exception ex)
			{
				MessageBox.Show(string.Format($"{ex.Message}"), "Can't create scheme");
			}
		}

		double percentage = 0;
		void backgroundWorker_DoWork(object sender, DoWorkEventArgs e)
		{
			String strErr = allIterations_DoWork();
			if (!string.IsNullOrEmpty(strErr)) MessageBox.Show(strErr, "VLPRectangle backgroundWorker_DoWork");
		}
		void backgroundWorker_ProgressChanged(object sender, ProgressChangedEventArgs e)
		{
			progressEvent?.Invoke(percentage);//don't use e.ProgressPercentage
		}
		void backgroundWorker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
		{
			completedEvent?.Invoke();
			worker = null;
		}

		public void changeMultiThread(bool isMultiThread)
		{
			optionsParallel.MaxDegreeOfParallelism = isMultiThread ? degreeOfParallelism : 1;
			GridIterator.optionsParallel = optionsParallel;
		}

		public void changeVisualParams(bool visualize1, int stepHeatMap1)
		{
			visualize = visualize1;
			stepHeatMap = stepHeatMap1;
		}

		public void allIterations()
		{
			if (worker != null) return;
			bCancelAllIterations = false;

			worker = new BackgroundWorker
			{
				WorkerReportsProgress = true,
				WorkerSupportsCancellation = true
			};
			worker.DoWork += backgroundWorker_DoWork;
			worker.ProgressChanged += backgroundWorker_ProgressChanged;
			worker.RunWorkerCompleted += backgroundWorker_RunWorkerCompleted;
			worker.RunWorkerAsync();
		}

		bool isSchemeUseInitialInterpolation(PlatformAndSchemeIndex platformScheme)
		{
			if (platformScheme.platrofm == PlatformEnum.CPU)
			{
				SchemeCPUEnum indexScheme = (SchemeCPUEnum)platformScheme.idxScheme;
				if (indexScheme == SchemeCPUEnum.CompleteReduction) return false;
				if (indexScheme == SchemeCPUEnum.VariablesSeparation) return false;
				if (indexScheme == SchemeCPUEnum.IncompleteReduction) return false;
				if (indexScheme == SchemeCPUEnum.MatrixProgonka) return false;
				if (indexScheme == SchemeCPUEnum.Marching) return false;
			}
			else if (platformScheme.platrofm == PlatformEnum.OCL)
			{
				SchemeOCLEnum indexScheme = (SchemeOCLEnum)platformScheme.idxScheme;
				if (indexScheme == SchemeOCLEnum.CompleteReductionOCL) return false;
				if (indexScheme == SchemeOCLEnum.VariablesSeparationOCL) return false;
				if (indexScheme == SchemeOCLEnum.IncompleteReductionOCL) return false;
				if (indexScheme == SchemeOCLEnum.MarchingOCL) return false;
			}
			else if (platformScheme.platrofm == PlatformEnum.CUDA)
			{
				SchemeCUDAEnum indexScheme = (SchemeCUDAEnum)platformScheme.idxScheme;
				if (indexScheme == SchemeCUDAEnum.VariablesSeparationCUDA) return false;
				if (indexScheme == SchemeCUDAEnum.FACRCUDA) return false;
				if (indexScheme == SchemeCUDAEnum.MarchingCUDA) return false;
			}

			return true;
		}

		void initScheme()
		{
			createScheme(platformScheme, methodParams);
			if (scheme == null) return ;

			T min = T.MaxValue, max = T.MinValue;
			scheme.initTopBottomBorders(stepX, stepY,pFuncBottom, pFuncTop, pFuncBoundary, ref min, ref max);
			scheme.initLeftRightBorders(stepX, stepY, pFuncLeft, pFuncRight, pFuncBoundary, ref min, ref max);

			if (isSchemeUseInitialInterpolation(platformScheme))
			{
				if (indexInterpolation == InterpolationEnumVLPRectangle.Mean) scheme.initInitialIterationMean((min + max) / T.CreateTruncating(2));
				else if (indexInterpolation == InterpolationEnumVLPRectangle.ArithmeticMean) scheme.initInitialIterationArithmeticMean();
				else if (indexInterpolation == InterpolationEnumVLPRectangle.Linear) scheme.initInitialIterationLinearInterpolation();
				else if (indexInterpolation == InterpolationEnumVLPRectangle.WeightLinear) scheme.initInitialIterationWeightLinearInterpolation();
			}

			if (methodParams.isCompareAnalytic && pFuncAnalytic != null)
			{
				(int, int) dims = scheme.getArrayDimensions();
				unDiff = new T[dims.Item1][];
				for (int i = 0; i < dims.Item1; i++) unDiff[i] = new T[dims.Item2];
			}
		}

		String allIterations_DoWork()
		{
			double prevPercent = 0;
			int iter = 0;
			elapsedIters = 0;
			BitmapSource bms, bmsDiff;
			Stopwatch stopWatch = new Stopwatch();

			progressHeaderEvent?.Invoke(Resources.strCalculating);
			stopWatch.Start();
			initScheme();

			if (scheme == null) return "";
			scheme.initAfterBoundariesAndInitialIterationInited();
			stopWatch.Stop();
			initTime = stopWatch.Elapsed.TotalSeconds;

			stopWatch.Restart();

			for (; iter < maxIters; iter++)
			{
				T deltaMaxIter = scheme.doIteration(iter);

				if (scheme.shouldReportProgress())
				{
					percentage = (iter * 100.0 / maxIters);
					if (percentage - prevPercent > 0.2)
					{
						worker.ReportProgress(0);//will be used above value
						prevPercent = percentage;
					}
				}

				if (visualize && (stepHeatMap != 0) && (iter % stepHeatMap == 0))
				{
					T min = T.MaxValue, max = T.MinValue;
					scheme.pointsMinMax(ref min, ref max);
					bms = scheme.createBitmap(new MinMaxF(float.CreateTruncating(min), float.CreateTruncating(max)), fCreateBitmap);
					if (bms == null) break;//in case low memory
					lstBitmap.Add(bms);

					if (methodParams.isCompareAnalytic && pFuncAnalytic != null)
					{
						scheme.calculateDifference(unDiff, stepX, stepY, pFuncAnalytic, ref fMinDiff, ref fMaxDiff, null, null);
						T coef = T.One;//diffCoef();
						MinMaxF minmax = new MinMaxF(float.CreateTruncating(fMinDiff * coef), float.CreateTruncating(fMaxDiff * coef));
						Adapter2D<float> adapter = new Adapter2D<float>(unDiff.GetUpperBound(0) + 1, unDiff[0].GetUpperBound(0) + 1, (i, j) => float.CreateTruncating(unDiff[i][j] * coef));
						bmsDiff = fCreateBitmapDiff(minmax, adapter);
						if (bmsDiff != null) lstBitmapDiff.Add(bmsDiff);
					}
				}

				if (bCancelAllIterations) break;
				if (deltaMaxIter < eps) break;
			}
			stopWatch.Stop();
			elapsedTime = stopWatch.Elapsed.TotalSeconds;

			if (!bCancelAllIterations)
			{
				T min = T.MaxValue, max = T.MinValue;
				scheme.pointsMinMax(ref min, ref max);//picture palette can change !
				bms = scheme.createBitmap(new MinMaxF(float.CreateTruncating(min), float.CreateTruncating(max)), fCreateBitmap);
				if (bms != null) lstBitmap.Add(bms);

				if (methodParams.isCompareAnalytic && pFuncAnalytic != null)
				{
					progressHeaderEvent?.Invoke(Resources.strComparing);
					scheme.calculateDifference(unDiff, stepX, stepY, pFuncAnalytic, ref fMinDiff, ref fMaxDiff, calculateDifferenceCanceled, reportProgress);
					T coef = T.One;//diffCoef();
					MinMaxF minmax = new MinMaxF(float.CreateTruncating(fMinDiff * coef), float.CreateTruncating(fMaxDiff * coef));
					Adapter2D<float> adapter = new Adapter2D<float>(unDiff.GetUpperBound(0) + 1, unDiff[0].GetUpperBound(0) + 1, (i, j) => float.CreateTruncating(unDiff[i][j] * coef));
					bmsDiff = fCreateBitmapDiff(minmax, adapter);
					if (bmsDiff != null) lstBitmapDiff.Add(bmsDiff);
					//Utils.printJaggedArray(unDiff, "{0,11:E3}", "unDiff");
				}
			}

			if (iter == 0) iter = 1;
			elapsedIters = iter;
			iterationsKind = scheme.iterationsKind();
			strElapsedInfo = scheme.getElapsedInfo();

			scheme.cleanup();
			scheme = null;
			GC.Collect();

			return null;
		}

		bool calculateDifferenceCanceled()
		{
			return bCancelAllIterations;
		}

		T diffCoef()
		{
			T log10 = T.Log10(T.Max(T.Abs(fMinDiff), T.Abs(fMaxDiff)));
			return log10 < T.Zero ? T.Pow(T.CreateTruncating(10), T.Abs(log10)) : T.One;
		}
		public void cancelAll()
		{
			bCancelAllIterations = true;
			scheme?.cancelIterations();//can be null if long time constructors
		}

		void reportProgress(double perc)//for complete reduction
		{
			percentage = perc;
			worker.ReportProgress(0);//will be used above value
		}

		public int getElapsedIters() {return elapsedIters;}

		public int getAllIters() {return maxIters;}
		public IterationsKind getIterationsKind() { return iterationsKind; }
		public double getInitTime() { return initTime;}
		public double getElapsedTime() {return elapsedTime;}

		public string getDeviation()
		{
			T deviation = T.Max(T.Abs(fMinDiff), T.Abs(fMaxDiff));
			string rc = String.Format(CultureInfo.InvariantCulture, "{0}", deviation);
			return rc;
		}

		public string getElapsedInfo() { return strElapsedInfo; }

		public void addProgressHandler(progressDelegateLPRectangle handler)
		{
			progressEvent += handler;
		}

		public void addProgressHeaderHandler(progressHeaderDelegateLPRectangle handler)
		{
			progressHeaderEvent += handler;
		}

		public void addCompletedHandler(completedDelegateLPRectangle handler)
		{
			completedEvent += handler;
		}

		public void removeProgressHandler(progressDelegateLPRectangle handler)
		{
			progressEvent -= handler;
		}

		public void removeProgressHeaderHandler(progressHeaderDelegateLPRectangle handler)
		{
			progressHeaderEvent -= handler;
		}

		public void removeCompletedHandler(completedDelegateLPRectangle handler)
		{
			completedEvent -= handler;
		}

		public string stringFunctionExpressionErrors()
		{
			return compiledFuncs.stringErrors();
		}
	}
}

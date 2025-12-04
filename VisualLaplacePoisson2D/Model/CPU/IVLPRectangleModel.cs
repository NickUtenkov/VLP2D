using System;
using System.Collections.Generic;
using System.Windows.Media.Imaging;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	public delegate void progressDelegateLPRectangle(double percent);
	public delegate void progressHeaderDelegateLPRectangle(string header);
	public delegate void completedDelegateLPRectangle();

	public enum SchemeCPUEnum : int
	{
		SimpleIteration = 0,
		SlidingIteration,
		SOR,//successive over relaxation
		Splitting,
		VarDir,
		PTM,
		GradientDescent,
		MinimumResidual,
		ConjugateGradient,
		BiconjugateGradient,
		Chebishev3Layers,
		MultiGrid,
		CompleteReduction,
		VariablesSeparation,
		IncompleteReduction,
		MatrixProgonka,
		Marching,
	}

	public enum SchemeOCLEnum : int
	{
		SimpleIterationOCL = 0,
		SlidingIterationOCL,
		SOROCL,//successive over relaxation
		SplittingOCL,
		VarDirOCL,
		MultiGridOCL,
		CompleteReductionOCL,
		VariablesSeparationOCL,
		IncompleteReductionOCL,
		MarchingOCL,
	}

	public enum SchemeCUDAEnum : int
	{
		SimpleIterationCUDA = 0,
		SlidingIterationCUDA,
		SORCUDA,//successive over relaxation
		SplittingCUDA,
		VarDirCUDA,
		MultiGridCUDA,
		CompleteReductionCUDA,
		VariablesSeparationCUDA,
		FACRCUDA,//IncompleteReduction
		MarchingCUDA
	}

	public enum PlatformEnum : int
	{
		CPU = 1,
		OCL,
		CUDA,
	}

	public enum VarSepMethodsEnum : int
	{
		FFT = 0,
		Progonka,
		Reduction,
		Marching
	}

	public enum CRMethodsEnum : int
	{
		SN = 0,//Samarskii-Nikolaev
		BM,//Buneman
	}

	public struct PlatformAndSchemeIndex
	{
		public PlatformEnum platrofm;
		public int idxScheme;

		public PlatformAndSchemeIndex(PlatformEnum platrofm, int idxScheme)
		{
			this.platrofm = platrofm;
			this.idxScheme = idxScheme;
		}
	}

	public enum InterpolationEnumVLPRectangle : int
	{
		Mean = 0,
		ArithmeticMean = 1,
		Linear = 2,
		WeightLinear = 3,
	}

	interface IVLPRectangleModel
	{
		void prepareCalculation(VLPRectangleParams pParams, InterpolationEnumVLPRectangle idxInterpolation, PlatformAndSchemeIndex platformScheme, List<BitmapSource> lstBitmap0, List<BitmapSource> lstBitmapDiff, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Func<MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmapDiff);
		void recalculateSteps(VLPRectangleParams pParams, PlatformAndSchemeIndex platformScheme, bool isVarSepProgonkaGPU, VarSepMethodsEnum varSepMethodCPU);
		void compileFunctions(VLPRectangleParams pParams);
		void setMethodParams(MethodsParams mParams);
		void changeMultiThread(bool isMulti);
		void changeVisualParams(bool visualize, int stepHeatMap);
		void setMaxIterations(int maxIterations);
		string stringFunctionExpressionErrors();
		void allIterations();
		void cancelAll();
		int getElapsedIters();
		int getAllIters();
		IterationsKind getIterationsKind();
		double getInitTime();
		double getElapsedTime();
		string getDeviation();
		string getElapsedInfo();
		int getMaxIterations(PlatformAndSchemeIndex platformScheme, VLPRectangleParams pParams, InterpolationEnumVLPRectangle idxInterpolation);
		void addProgressHandler(progressDelegateLPRectangle handler);
		void addProgressHeaderHandler(progressHeaderDelegateLPRectangle handler);
		void addCompletedHandler(completedDelegateLPRectangle handler);
		void removeProgressHandler(progressDelegateLPRectangle handler);
		void removeProgressHeaderHandler(progressHeaderDelegateLPRectangle handler);
		void removeCompletedHandler(completedDelegateLPRectangle handler);
	}
}

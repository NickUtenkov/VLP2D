using Cloo;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Input;
using VLP2D.Common;
using VLP2D.Model;
using VLP2D.Properties;

namespace VLP2D.ViewModel
{
	public class VLPRectangleInputVM : ObservableObject, IVLPRectangleInput
	{
		IVLPRectangleOutput output = null;
		public List<VLPRectangleParams> listParams { get; } = new List<VLPRectangleParams>();
		public List<String> listInterpolations { get; } = new List<string> { Resources.strMean, Resources.strArithMean, Resources.strPiecewise, Resources.strPiecewiseWeight };

		public List<String> listCPUSchemes { get; } = new List<string> { Resources.strSimpleIteration, Resources.strSlidingIteration, Resources.strSOR, Resources.strSplitting,
			Resources.strVarDir, Resources.strPTM, Resources.strGradDesc, Resources.strMinRes, Resources.strConjGrad, Resources.strBiconjGrad, Resources.strChebish3,
			Resources.strMultiGridVC, Resources.strCR, Resources.strFA, Resources.strFACR, Resources.strMatrProgonka, Resources.strMarchingVector,};

		public List<String> listOCLSchemes { get; } = new List<string> { Resources.strSimpleIteration, Resources.strSlidingIteration, Resources.strSOR, Resources.strSplitting,
			Resources.strVarDir, Resources.strMultiGridVC, Resources.strCR, Resources.strFA, Resources.strFACR, Resources.strMarchingVector};

		public List<String> listCUDASchemes { get; } = new List<string> { Resources.strSimpleIteration,Resources.strSlidingIteration, Resources.strSOR, Resources.strSplitting,
			Resources.strVarDir, Resources.strMultiGridVC, Resources.strCR, Resources.strFA, Resources.strFACR, Resources.strMarchingVector};

		public List<UtilsCL.PlatformDevice> listOpenCLDevices { get; } = new List<UtilsCL.PlatformDevice>();
		public List<UtilsCU.DeviceCU> listCUDADevices { get; } = new List<UtilsCU.DeviceCU>();
		public List<String> listPrecisions { get; } = new List<string> { /*Resources.strPrecisionHalf,*/ Resources.strPrecisionFloat, Resources.strPrecisionDouble, Resources.strPrecisionDD128, Resources.strPrecisionQD256 };
		public List<String> listVarSepMethods { get; } = new List<string> { Resources.strFFT, Resources.strProgonka, Resources.strReduction, Resources.strMarchingScalar };
		public List<String> listCRMethods { get; } = new List<string> { Resources.strMethodSN, Resources.strMethodBM };
		public VLPRectangleParams currentParams = null;
		int[] paramsL = new int[1];
		int savedLParam = int.MaxValue;
		bool isPrepareCalculationAllowed = true;

		public VLPRectangleInputVM()
		{
			listOpenCLDevices.AddRange(UtilsCL.getGPUDevices());
			listCUDADevices.AddRange(UtilsCU.getCUDADevices());
			readInputData();
		}

		public ICommand allIterationsCommand
		{
			get { return new DelegateCommand(() => output?.allIterations()); }
		}

		public VLPRectangleParams getInputParameters()
		{
			return currentParams;
		}

		public bool shouldVisualize()
		{
			return isVisualization;
		}

		public bool shouldSaveAnimatedGIF()
		{
			return isSavingGIF;
		}

		public bool shouldMultiThread()
		{
			return isMultiThread;
		}

		public bool shouldUseChebysh()
		{
			return isChebysh;
		}

		public bool shouldUseJordan()
		{
			return isJordan;
		}

		public bool shouldUseBiconjugateStabilized()
		{
			return isBiconjugateStabilized;
		}

		public bool shouldUseVarSepProgonka()
		{
			return isVarSepProgonka;
		}

		public VarSepMethodsEnum varSepMethod()
		{
			return varSepIndex;
		}

		public CRMethodsEnum crMethod()
		{
			return crIndex;
		}

		public bool shouldUseCompareAnalytic()
		{
			return showCompareAnalytic && isCompareAnalytic;
		}

		public int getLParam()
		{
			if (idxLParam >= 0 && idxLParam < paramsL.Length) return paramsL[idxLParam];
			return 0;
		}

		public PlatformOCL oclPlatform()
		{
			return isOpenCLComboBoxVisible ? listOpenCLDevices[deviceOCLSelectedIndex].platform : null;
		}

		public DeviceOCL oclDevice()
		{
			return isOpenCLComboBoxVisible ? listOpenCLDevices[deviceOCLSelectedIndex].device : null;
		}

		public int cudaDevice()
		{
			return isCUDAComboBoxVisible ? deviceCUDASelectedIndex : -1;
		}

		public InterpolationEnumVLPRectangle getInterpolationIndex()
		{
			return interpolationIndex;
		}

		public int precisionIndex()
		{
			return precisionSelectedIndex;
		}

		public PlatformAndSchemeIndex getPlatformAndSchemeIndex()
		{
			if (isOpenCLCheckBoxChecked) return new PlatformAndSchemeIndex(PlatformEnum.OCL, (int)schemeOCLIndex);
			else if (isCUDACheckBoxChecked) return new PlatformAndSchemeIndex(PlatformEnum.CUDA, (int)schemeCUDAIndex);
			return new PlatformAndSchemeIndex(PlatformEnum.CPU, (int)schemeCPUIndex);
		}

		public void setOutput(IVLPRectangleOutput inValue)
		{
			output = inValue;
		}

		bool _showCalculate = false;
		public bool showCalculate
		{
			get { return _showCalculate; }
			set
			{
				if (_showCalculate == value) return;
				_showCalculate = value;
				RaisePropertyChangedEvent("showCalculate");
			}
		}

		int _taskSelectedIndex = -1;
		public int taskSelectedIndex
		{
			get { return _taskSelectedIndex; }
			set
			{
				if (value >= 0 && value < listParams.Count)
				{
					_taskSelectedIndex = value;
					currentParams = listParams[value];
					RaisePropertyChangedEvent("taskSelectedIndex");
					readTaskParams();
					showCompareAnalytic = currentParams.funcAnalytic != null;
					if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
					output?.compileModelFunctions(getInputParameters());
					output?.doPrepareCalculation();
					showCalculate = true;
					updateMethodParamsVisibility();
				}
			}
		}
		
		void readTaskParams()
		{
			TaskParams taskParams = TaskParams.restore(currentParams.name);
			if (taskParams != null)
			{
				isPrepareCalculationAllowed = false;

				isOpenCLCheckBoxChecked = taskParams.isOpenCLCheckBoxChecked;
				isCUDACheckBoxChecked = taskParams.isCUDACheckBoxChecked;
				if (isOpenCLCheckBoxChecked) schemeOCLIndex = (SchemeOCLEnum)taskParams.idxScheme;
				else if (isCUDACheckBoxChecked) schemeCUDAIndex = (SchemeCUDAEnum)taskParams.idxScheme;
				else schemeCPUIndex = (SchemeCPUEnum)taskParams.idxScheme;
				deviceOCLSelectedIndex = taskParams.idxDeviceOCL;
				deviceCUDASelectedIndex = taskParams.idxDeviceCUDA;

				interpolationIndex = (InterpolationEnumVLPRectangle)taskParams.idxInterpol;
				isVarSepProgonka = taskParams.isVarSepProgonka;
				varSepIndex = taskParams.varSepMethod;
				crIndex = taskParams.crMethod;
				isBiconjugateStabilized = taskParams.isBiconjugateStabilized;
				isJordan = taskParams.isJordan;
				isChebysh = taskParams.isChebysh;
				precisionSelectedIndex = taskParams.precision;

				savedLParam = isLParamMethod() ? taskParams.paramL : int.MaxValue;

				isPrepareCalculationAllowed = true;
			}
		}

		InterpolationEnumVLPRectangle _interpolationIndex = InterpolationEnumVLPRectangle.Mean;
		public InterpolationEnumVLPRectangle interpolationIndex
		{
			get { return _interpolationIndex; }
			set
			{
				_interpolationIndex = value;
				RaisePropertyChangedEvent("interpolationIndex");
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		VarSepMethodsEnum _varSepIndex = VarSepMethodsEnum.FFT;
		public VarSepMethodsEnum varSepIndex
		{
			get { return _varSepIndex; }
			set
			{
				_varSepIndex = value;
				RaisePropertyChangedEvent("varSepIndex");
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		CRMethodsEnum _crIndex = CRMethodsEnum.SN;
		public CRMethodsEnum crIndex
		{
			get { return _crIndex; }
			set
			{
				_crIndex = value;
				RaisePropertyChangedEvent("crIndex");
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		bool _showInterpolation = true;
		public bool showInterpolation
		{
			get { return _showInterpolation; }
			set
			{
				if (_showInterpolation == value) return;
				_showInterpolation = value;
				RaisePropertyChangedEvent("showInterpolation");
			}
		}

		void updateMethodParamsVisibility()
		{
			if (!isOpenCLCheckBoxChecked && !isCUDACheckBoxChecked)
			{
				showChebysh = (schemeCPUIndex == SchemeCPUEnum.PTM || schemeCPUIndex == SchemeCPUEnum.SimpleIteration);
				showJordan = schemeCPUIndex == SchemeCPUEnum.VarDir;
				showBiconjugateStabilized = schemeCPUIndex == SchemeCPUEnum.BiconjugateGradient;
				showMethodHeader = (schemeCPUIndex == SchemeCPUEnum.CompleteReduction || schemeCPUIndex == SchemeCPUEnum.VariablesSeparation);
				showVarSepCombo = schemeCPUIndex == SchemeCPUEnum.VariablesSeparation;
				showCRCombo = schemeCPUIndex == SchemeCPUEnum.CompleteReduction;
				showVarSepProgonka = false;

				showInterpolation = !(
					schemeCPUIndex == SchemeCPUEnum.CompleteReduction ||
					schemeCPUIndex == SchemeCPUEnum.VariablesSeparation ||
					schemeCPUIndex == SchemeCPUEnum.IncompleteReduction ||
					schemeCPUIndex == SchemeCPUEnum.MatrixProgonka ||
					schemeCPUIndex == SchemeCPUEnum.Marching ||
					boundaryNull());
				if (!showInterpolation) interpolationIndex = InterpolationEnumVLPRectangle.Mean;
				showLParamSlider = !isOpenCLComboBoxVisible && schemeCPUIndex == SchemeCPUEnum.Marching || schemeCPUIndex == SchemeCPUEnum.IncompleteReduction;
				if (showLParamSlider && (schemeCPUIndex == SchemeCPUEnum.Marching)) resetMarchingLParam();
				if (showLParamSlider && (schemeCPUIndex == SchemeCPUEnum.IncompleteReduction)) resetFACRLParam(true);
			}
			else if (isOpenCLCheckBoxChecked)
			{
				showChebysh = schemeOCLIndex == SchemeOCLEnum.SimpleIterationOCL;
				showJordan = schemeOCLIndex == SchemeOCLEnum.VarDirOCL;
				showBiconjugateStabilized = false;
				showVarSepProgonka = schemeOCLIndex == SchemeOCLEnum.VariablesSeparationOCL;
				showMethodHeader = false;
				showVarSepCombo = false;
				showCRCombo = false;

				showInterpolation = !(
					schemeOCLIndex == SchemeOCLEnum.CompleteReductionOCL ||
					schemeOCLIndex == SchemeOCLEnum.VariablesSeparationOCL ||
					schemeOCLIndex == SchemeOCLEnum.IncompleteReductionOCL ||
					schemeOCLIndex == SchemeOCLEnum.MarchingOCL ||
					boundaryNull());
				if (!showInterpolation) interpolationIndex = InterpolationEnumVLPRectangle.Mean;
				showLParamSlider = (schemeOCLIndex == SchemeOCLEnum.IncompleteReductionOCL) || (schemeOCLIndex == SchemeOCLEnum.MarchingOCL);
				if (showLParamSlider && (schemeOCLIndex == SchemeOCLEnum.IncompleteReductionOCL)) resetFACRLParam(false);
				if (showLParamSlider && (schemeOCLIndex == SchemeOCLEnum.MarchingOCL)) resetMarchingLParam();
			}
			else if (isCUDACheckBoxChecked)
			{
				showChebysh = schemeCUDAIndex == SchemeCUDAEnum.SimpleIterationCUDA;
				showJordan = schemeCUDAIndex == SchemeCUDAEnum.VarDirCUDA;
				showBiconjugateStabilized = false;
				showVarSepProgonka = schemeCUDAIndex == SchemeCUDAEnum.VariablesSeparationCUDA;
				showMethodHeader = false;
				showVarSepCombo = false;
				showCRCombo = false;

				showInterpolation = !(
					schemeCUDAIndex == SchemeCUDAEnum.CompleteReductionCUDA ||
					schemeCUDAIndex == SchemeCUDAEnum.VariablesSeparationCUDA ||
					schemeCUDAIndex == SchemeCUDAEnum.FACRCUDA ||
					schemeCUDAIndex == SchemeCUDAEnum.MarchingCUDA ||
					boundaryNull());
				if (!showInterpolation) interpolationIndex = InterpolationEnumVLPRectangle.Mean;
				showLParamSlider = (schemeCUDAIndex == SchemeCUDAEnum.FACRCUDA) || (schemeCUDAIndex == SchemeCUDAEnum.MarchingCUDA);
				if (showLParamSlider && (schemeCUDAIndex == SchemeCUDAEnum.FACRCUDA)) resetFACRLParam(false);
				if (showLParamSlider && (schemeCUDAIndex == SchemeCUDAEnum.MarchingCUDA)) resetMarchingLParam();
			}
			isDirectOrFixedIterationsMethod = methodIsDirectOrFixedIterations();
		}

		bool boundaryNull()
		{
			bool b1 = string.IsNullOrEmpty(currentParams.funcLeft) || currentParams.funcLeft.Equals("0");
			bool b2 = string.IsNullOrEmpty(currentParams.funcRight) || currentParams.funcRight.Equals("0");
			bool b3 = string.IsNullOrEmpty(currentParams.funcTop) || currentParams.funcTop.Equals("0");
			bool b4 = string.IsNullOrEmpty(currentParams.funcBottom) || currentParams.funcBottom.Equals("0");
			bool b5 = string.IsNullOrEmpty(currentParams.funcBoundary) || currentParams.funcBoundary.Equals("0");

			return (b1 && b2 && b3 && b4) && b5;
		}

		SchemeCPUEnum _schemeCPUIndex = SchemeCPUEnum.SimpleIteration;//temporary value, see init
		public SchemeCPUEnum schemeCPUIndex
		{
			get { return _schemeCPUIndex; }
			set
			{
				_schemeCPUIndex = value;
				RaisePropertyChangedEvent("schemeCPUIndex");
				RaisePropertyChangedEvent("isStepVisible");
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();

				string str = listCPUSchemes[(int)schemeCPUIndex];
				syncCUDAIndex(str);
				syncOCLIndex(str);

				updateMethodParamsVisibility();
			}
		}

		SchemeOCLEnum _schemeOCLIndex = SchemeOCLEnum.SimpleIterationOCL;//temporary value, see init; MarchingOCL
		public SchemeOCLEnum schemeOCLIndex
		{
			get { return _schemeOCLIndex; }
			set
			{
				_schemeOCLIndex = value;
				RaisePropertyChangedEvent("schemeOCLIndex");
				RaisePropertyChangedEvent("isStepVisible");
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();

				string str = listOCLSchemes[(int)schemeOCLIndex];
				syncCUDAIndex(str);
				syncCPUIndex(str);

				updateMethodParamsVisibility();
			}
		}

		SchemeCUDAEnum _schemeCUDAIndex = SchemeCUDAEnum.SimpleIterationCUDA;//temporary value, see init
		public SchemeCUDAEnum schemeCUDAIndex
		{
			get { return _schemeCUDAIndex; }
			set
			{
				_schemeCUDAIndex = value;
				RaisePropertyChangedEvent("schemeCUDAIndex");
				RaisePropertyChangedEvent("isStepVisible");
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();

				string str = listCUDASchemes[(int)schemeCUDAIndex];
				syncCPUIndex(str);
				syncOCLIndex(str);

				updateMethodParamsVisibility();
			}
		}

		void syncCUDAIndex(string str)
		{
			int idx = listCUDASchemes.FindIndex(x => x == str);
			if (idx != -1)
			{
				_schemeCUDAIndex = (SchemeCUDAEnum)idx;
				RaisePropertyChangedEvent("schemeCUDAIndex");
			}
		}

		void syncCPUIndex(string str)
		{
			int idx = listCPUSchemes.FindIndex(x => x == str);
			if (idx != -1)
			{
				_schemeCPUIndex = (SchemeCPUEnum)idx;
				RaisePropertyChangedEvent("schemeCPUIndex");
			}
		}

		void syncOCLIndex(string str)
		{
			int idx = listOCLSchemes.FindIndex(x => x == str);
			if (idx != -1)
			{
				_schemeOCLIndex = (SchemeOCLEnum)idx;
				RaisePropertyChangedEvent("schemeOCLIndex");
			}
		}

		bool _isVisualization = false;
		public bool isVisualization
		{
			get { return _isVisualization; }
			set
			{
				if (_isVisualization == value) return;
				_isVisualization = value;
				RaisePropertyChangedEvent("isVisualization");
				RaisePropertyChangedEvent("isStepVisible");
				output?.changeModelVisualParams(isVisualization, getVisualStep());
			}
		}

		bool _isSavingGIF = false;
		public bool isSavingGIF
		{
			get { return _isSavingGIF; }
			set
			{
				if (_isSavingGIF == value) return;
				_isSavingGIF = value;
				RaisePropertyChangedEvent("isSavingGIF");
			}
		}

		bool _isMultiThread = true;
		public bool isMultiThread
		{
			get { return _isMultiThread; }
			set
			{
				if (_isMultiThread == value) return;
				_isMultiThread = value;
				RaisePropertyChangedEvent("isMultiThread");
				output?.changeModelMultiThread(_isMultiThread);
			}
		}

		bool _showMultiThread = true;
		public bool showMultiThread
		{
			get { return _showMultiThread; }
			set
			{
				if (_showMultiThread == value) return;
				_showMultiThread = value;
				RaisePropertyChangedEvent("showMultiThread");
			}
		}

		bool _isChebysh = true;
		public bool isChebysh
		{
			get { return _isChebysh; }
			set
			{
				if (_isChebysh == value) return;
				_isChebysh = value;
				RaisePropertyChangedEvent("isChebysh");
				RaisePropertyChangedEvent("isStepVisible");
				output?.setModelMethodsParams();
				isDirectOrFixedIterationsMethod = methodIsDirectOrFixedIterations();
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				countHeatMaps = (countIterations / stepVis).ToString();
			}
		}

		bool _showChebysh = false;
		public bool showChebysh
		{
			get { return _showChebysh; }
			set
			{
				if (_showChebysh == value) return;
				_showChebysh = value;
				RaisePropertyChangedEvent("showChebysh");
			}
		}

		bool _isJordan = true;
		public bool isJordan
		{
			get { return _isJordan; }
			set
			{
				if (_isJordan == value) return;
				_isJordan = value;
				RaisePropertyChangedEvent("isJordan");
				RaisePropertyChangedEvent("isStepVisible");
				output?.setModelMethodsParams();
				isDirectOrFixedIterationsMethod = methodIsDirectOrFixedIterations();
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				countHeatMaps = (countIterations / stepVis).ToString();
			}
		}

		bool _showJordan = false;
		public bool showJordan
		{
			get { return _showJordan; }
			set
			{
				if (_showJordan == value) return;
				_showJordan = value;
				RaisePropertyChangedEvent("showJordan");
			}
		}

		bool _isBiconjugateStabilized = true;
		public bool isBiconjugateStabilized
		{
			get { return _isBiconjugateStabilized; }
			set
			{
				if (_isBiconjugateStabilized == value) return;
				_isBiconjugateStabilized = value;
				RaisePropertyChangedEvent("isBiconjugateStabilized");
				output?.setModelMethodsParams();
			}
		}

		bool _showBiconjugateStabilized = false;
		public bool showBiconjugateStabilized
		{
			get { return _showBiconjugateStabilized; }
			set
			{
				if (_showBiconjugateStabilized == value) return;
				_showBiconjugateStabilized = value;
				RaisePropertyChangedEvent("showBiconjugateStabilized");
			}
		}

		bool _isVarSepProgonka = false;
		public bool isVarSepProgonka
		{
			get { return _isVarSepProgonka; }
			set
			{
				if (_isVarSepProgonka == value) return;
				_isVarSepProgonka = value;
				RaisePropertyChangedEvent("isVarSepProgonka");
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		bool _showVarSepProgonka = false;
		public bool showVarSepProgonka
		{
			get { return _showVarSepProgonka; }
			set
			{
				if (_showVarSepProgonka == value) return;
				_showVarSepProgonka = value;
				RaisePropertyChangedEvent("showVarSepProgonka");
			}
		}

		bool _showMethodHeader = false;
		public bool showMethodHeader
		{
			get { return _showMethodHeader; }
			set
			{
				if (_showMethodHeader == value) return;
				_showMethodHeader = value;
				RaisePropertyChangedEvent("showMethodHeader");
			}
		}

		bool _showVarSepCombo = false;
		public bool showVarSepCombo
		{
			get { return _showVarSepCombo; }
			set
			{
				if (_showVarSepCombo == value) return;
				_showVarSepCombo = value;
				RaisePropertyChangedEvent("showVarSepCombo");
			}
		}

		bool _showCRCombo = false;
		public bool showCRCombo
		{
			get { return _showCRCombo; }
			set
			{
				if (_showCRCombo == value) return;
				_showCRCombo = value;
				RaisePropertyChangedEvent("showCRCombo");
			}
		}

		bool _isCompareAnalytic = false;
		public bool isCompareAnalytic
		{
			get { return _isCompareAnalytic; }
			set
			{
				if (_isCompareAnalytic == value) return;
				_isCompareAnalytic = value;
				RaisePropertyChangedEvent("isCompareAnalytic");
				output?.setModelMethodsParams();
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		bool _showCompareAnalytic = false;
		public bool showCompareAnalytic
		{
			get { return _showCompareAnalytic; }
			set
			{
				if (_showCompareAnalytic == value) return;
				_showCompareAnalytic = value;
				RaisePropertyChangedEvent("showCompareAnalytic");
			}
		}

		double _sliderLParamTicksMax = 0;
		public double sliderLParamTicksMax
		{
			get { return _sliderLParamTicksMax; }
			set
			{
				_sliderLParamTicksMax = value;
				RaisePropertyChangedEvent("sliderLParamTicksMax");
			}
		}

		double _sliderLParamTicksMin = 0;
		public double sliderLParamTicksMin
		{
			get { return _sliderLParamTicksMin; }
			set
			{
				_sliderLParamTicksMin = value;
				RaisePropertyChangedEvent("sliderLParamTicksMin");
			}
		}

		int _idxLParam = 99;
		public int idxLParam
		{
			get { return _idxLParam; }
			set
			{
				_idxLParam = value;
				RaisePropertyChangedEvent("idxLParam");
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		bool _showLParamSlider = false;
		public bool showLParamSlider
		{
			get { return _showLParamSlider; }
			set
			{
				if (_showLParamSlider == value) return;
				_showLParamSlider = value;
				RaisePropertyChangedEvent("showLParamSlider");
			}
		}

		bool isLParamMethod()
		{
			if (!isOpenCLCheckBoxChecked && !isCUDACheckBoxChecked && schemeCPUIndex == SchemeCPUEnum.Marching) return true;
			if (isOpenCLCheckBoxChecked && schemeOCLIndex == SchemeOCLEnum.MarchingOCL) return true;
			if (isCUDACheckBoxChecked && schemeCUDAIndex == SchemeCUDAEnum.MarchingCUDA) return true;

			if (!isOpenCLCheckBoxChecked && !isCUDACheckBoxChecked && schemeCPUIndex == SchemeCPUEnum.IncompleteReduction) return true;
			if (isOpenCLCheckBoxChecked && schemeOCLIndex == SchemeOCLEnum.IncompleteReductionOCL) return true;
			if (isCUDACheckBoxChecked && schemeCUDAIndex == SchemeCUDAEnum.FACRCUDA) return true;

			return false;
		}

		void resetMarchingLParam()
		{
			if (getInputParameters() != null)
			{
				paramsL = UtilsLParam.getMarchingLParamArray(getInputParameters().cXSegments);
				if (paramsL.Length == 0) paramsL = new int[1];
				sliderLParamTicksMax = paramsL.Length - 1;
				sliderLParamTicksMin = sliderLParamTicksMax - 3 - precisionSelectedIndex;
				if (sliderLParamTicksMin < 0) sliderLParamTicksMin = 0;
				int idx = Array.FindIndex(paramsL, x => x == savedLParam);
				idxLParam = (idx != -1) ? idx : paramsL.Length - 1;
			}
			else paramsL = new int[1];
		}

		void resetFACRLParam(bool useCXSegments)
		{
			if (getInputParameters() != null)
			{
				paramsL = UtilsLParam.getFACRLParamArray(useCXSegments ? getInputParameters().cXSegments : getInputParameters().cYSegments);
				sliderLParamTicksMax = paramsL.Length - 1;
				sliderLParamTicksMin = 0;
				int idx = Array.FindIndex(paramsL, x => x == savedLParam);
				idxLParam = (idx != -1) ? idx : paramsL.Length - 1;
			}
			else paramsL = new int[1];
		}

		bool _isOpenCLCheckBoxChecked = false;
		public bool isOpenCLCheckBoxChecked
		{
			get { return _isOpenCLCheckBoxChecked; }
			set
			{
				if (_isOpenCLCheckBoxChecked == value) return;
				_isOpenCLCheckBoxChecked = value;
				isOpenCLComboBoxVisible = isOpenCLCheckBoxChecked;
				RaisePropertyChangedEvent("isOpenCLCheckBoxChecked");
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
				if (_isOpenCLCheckBoxChecked) isCUDACheckBoxChecked = false;
				updateMethodParamsVisibility();
			}
		}
		public bool isOpenCLUsing() => isOpenCLCheckBoxChecked;

		bool _isOpenCLComboBoxVisible = false;
		public bool isOpenCLComboBoxVisible
		{
			get { return _isOpenCLComboBoxVisible; }
			set
			{
				if (_isOpenCLComboBoxVisible == value) return;
				_isOpenCLComboBoxVisible = value;
				if (deviceOCLSelectedIndex == -1) deviceOCLSelectedIndex = 0;//select first item only first time
				RaisePropertyChangedEvent("isOpenCLComboBoxVisible");
			}
		}

		int _deviceOCLSelectedIndex = -1;
		public int deviceOCLSelectedIndex
		{
			get { return _deviceOCLSelectedIndex; }
			set
			{
				_deviceOCLSelectedIndex = value;
				RaisePropertyChangedEvent("deviceOCLSelectedIndex");
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		bool _isCUDACheckBoxChecked = false;
		public bool isCUDACheckBoxChecked
		{
			get { return _isCUDACheckBoxChecked; }
			set
			{
				if (_isCUDACheckBoxChecked == value) return;
				_isCUDACheckBoxChecked = value;
				isCUDAComboBoxVisible = isCUDACheckBoxChecked;
				RaisePropertyChangedEvent("isCUDACheckBoxChecked");
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
				if (_isCUDACheckBoxChecked) isOpenCLCheckBoxChecked = false;
				updateMethodParamsVisibility();
			}
		}
		public bool isCUDAUsing() => isCUDACheckBoxChecked;

		bool _isCUDAComboBoxVisible = false;
		public bool isCUDAComboBoxVisible
		{
			get { return _isCUDAComboBoxVisible; }
			set
			{
				if (_isCUDAComboBoxVisible == value) return;
				_isCUDAComboBoxVisible = value;
				if (deviceCUDASelectedIndex == -1) deviceCUDASelectedIndex = 0;//select first item only first time
				RaisePropertyChangedEvent("isCUDAComboBoxVisible");
			}
		}

		int _deviceCUDASelectedIndex = -1;
		public int deviceCUDASelectedIndex
		{
			get { return _deviceCUDASelectedIndex; }
			set
			{
				_deviceCUDASelectedIndex = value;
				RaisePropertyChangedEvent("deviceCUDASelectedIndex");
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		public int deviceOCLIndex()
		{
			return deviceOCLSelectedIndex;
		}

		public int deviceCUDAIndex()
		{
			return deviceCUDASelectedIndex;
		}

		int _precisionSelectedIndex = 2;
		public int precisionSelectedIndex
		{
			get { return _precisionSelectedIndex; }
			set
			{
				if (_precisionSelectedIndex == value) return;
				_precisionSelectedIndex = value;
				RaisePropertyChangedEvent("precisionSelectedIndex");
				output?.changeModelPrecision(_precisionSelectedIndex);
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				if (showLParamSlider)
				{
					if (!isOpenCLCheckBoxChecked && !isCUDACheckBoxChecked && schemeCPUIndex == SchemeCPUEnum.Marching) resetMarchingLParam();
					if (isOpenCLCheckBoxChecked && schemeOCLIndex == SchemeOCLEnum.MarchingOCL) resetMarchingLParam();
					if (isCUDACheckBoxChecked && schemeCUDAIndex == SchemeCUDAEnum.MarchingCUDA) resetMarchingLParam();
				}
				output?.compileModelFunctions(getInputParameters());
				if (output != null) countIterations = output.getModelMaxIterations(getPlatformAndSchemeIndex(), currentParams, getInterpolationIndex());
				if (isPrepareCalculationAllowed) output?.doPrepareCalculation();
			}
		}

		bool _isDirectOrFixedIterationsMethod = false;
		public bool isDirectOrFixedIterationsMethod
		{
			get { return _isDirectOrFixedIterationsMethod; }
			set
			{
				if (_isDirectOrFixedIterationsMethod == value) return;
				_isDirectOrFixedIterationsMethod = value;
				RaisePropertyChangedEvent("isDirectOrFixedIterationsMethod");
			}
		}

		int _countIterations = 1;
		public int countIterations
		{
			get { return _countIterations; }
			set
			{
				output?.setModelIterations(value);
				if (_countIterations == value) return;
				_countIterations = value;
				RaisePropertyChangedEvent("countIterations");
				countHeatMaps = (countIterations / stepVis).ToString();
			}
		}

		public int maxIterations()
		{
			return countIterations;
		}

		string _countHeatMaps = "";
		public string countHeatMaps
		{
			get { return _countHeatMaps; }
			set
			{
				//if (_countHeatMaps == value) return;
				if (!methodIsFixedIterations()) _countHeatMaps = value;
				else _countHeatMaps = "?";
				RaisePropertyChangedEvent("countHeatMaps");
			}
		}

		public int getVisualStep()
		{
			return isVisibleStepVisualization() ? stepVis : 0;
		}

		int _stepVis = 5;
		public int stepVis
		{
			get { return _stepVis; }
			set
			{
				if (_stepVis == value) return;
				_stepVis = value > 0 ? value : 1;
				RaisePropertyChangedEvent("stepVis");
				countHeatMaps = (countIterations / stepVis).ToString();
			}
		}

		public bool methodIsFixedIterations()
		{
			if (isOpenCLCheckBoxChecked)
			{
				if (schemeOCLIndex == SchemeOCLEnum.SimpleIterationOCL && isChebysh) return true;
				if (schemeOCLIndex == SchemeOCLEnum.VarDirOCL && isJordan) return true;
			}
			else if (isCUDACheckBoxChecked)
			{
				if (schemeCUDAIndex == SchemeCUDAEnum.SimpleIterationCUDA && isChebysh) return true;
				if (schemeCUDAIndex == SchemeCUDAEnum.VarDirCUDA && isJordan) return true;
			}
			else if (!isOpenCLCheckBoxChecked && !isCUDACheckBoxChecked)
			{
				if (schemeCPUIndex == SchemeCPUEnum.SimpleIteration && isChebysh) return true;
				if (schemeCPUIndex == SchemeCPUEnum.VarDir && isJordan) return true;
				if (schemeCPUIndex == SchemeCPUEnum.PTM && isChebysh) return true;
			}

			return false;
		}

		public bool methodIsDirectOrFixedIterations()
		{
			if (isOpenCLCheckBoxChecked)
			{
				if (schemeOCLIndex == SchemeOCLEnum.SimpleIterationOCL && !isChebysh) return false;
				if (schemeOCLIndex == SchemeOCLEnum.SlidingIterationOCL) return false;
				if (schemeOCLIndex == SchemeOCLEnum.SOROCL) return false;
				if (schemeOCLIndex == SchemeOCLEnum.SplittingOCL) return false;
				if (schemeOCLIndex == SchemeOCLEnum.VarDirOCL && !isJordan) return false;
				if (schemeOCLIndex == SchemeOCLEnum.MultiGridOCL) return false;
			}
			else if (isCUDACheckBoxChecked)
			{
				if (schemeCUDAIndex == SchemeCUDAEnum.SimpleIterationCUDA && !isChebysh) return false;
				if (schemeCUDAIndex == SchemeCUDAEnum.SlidingIterationCUDA) return false;
				if (schemeCUDAIndex == SchemeCUDAEnum.SORCUDA) return false;
				if (schemeCUDAIndex == SchemeCUDAEnum.SplittingCUDA) return false;
				if (schemeCUDAIndex == SchemeCUDAEnum.VarDirCUDA && !isJordan) return false;
				if (schemeCUDAIndex == SchemeCUDAEnum.MultiGridCUDA) return false;
			}
			else if(!isOpenCLCheckBoxChecked && !isCUDACheckBoxChecked)
			{
				if (schemeCPUIndex == SchemeCPUEnum.SimpleIteration && !isChebysh) return false;
				if (schemeCPUIndex == SchemeCPUEnum.SlidingIteration) return false;
				if (schemeCPUIndex == SchemeCPUEnum.SOR) return false;
				if (schemeCPUIndex == SchemeCPUEnum.Splitting) return false;
				if (schemeCPUIndex == SchemeCPUEnum.VarDir && !isJordan) return false;
				if (schemeCPUIndex == SchemeCPUEnum.PTM && !isChebysh) return false;
				if (schemeCPUIndex == SchemeCPUEnum.GradientDescent) return false;
				if (schemeCPUIndex == SchemeCPUEnum.MinimumResidual) return false;
				if (schemeCPUIndex == SchemeCPUEnum.ConjugateGradient) return false;
				if (schemeCPUIndex == SchemeCPUEnum.BiconjugateGradient) return false;
				if (schemeCPUIndex == SchemeCPUEnum.Chebishev3Layers) return false;
				if (schemeCPUIndex == SchemeCPUEnum.MultiGrid) return false;
			}

			return true;
		}

		public bool isStepVisible
		{
			get { return isVisualization && isVisibleStepVisualization(); }
		}

		bool isVisibleStepVisualization()
		{
			if (isOpenCLCheckBoxChecked)
			{
				if (schemeOCLIndex == SchemeOCLEnum.SimpleIterationOCL) return true;
				if (schemeOCLIndex == SchemeOCLEnum.SlidingIterationOCL) return true;
				if (schemeOCLIndex == SchemeOCLEnum.SOROCL) return true;
				if (schemeOCLIndex == SchemeOCLEnum.SplittingOCL) return true;
				if (schemeOCLIndex == SchemeOCLEnum.VarDirOCL) return true;
				if (schemeOCLIndex == SchemeOCLEnum.MultiGridOCL) return true;
			}
			else if (isCUDACheckBoxChecked)
			{
				if (schemeCUDAIndex == SchemeCUDAEnum.SimpleIterationCUDA) return true;
				if (schemeCUDAIndex == SchemeCUDAEnum.SlidingIterationCUDA) return true;
				if (schemeCUDAIndex == SchemeCUDAEnum.SORCUDA) return true;
				if (schemeCUDAIndex == SchemeCUDAEnum.SplittingCUDA) return true;
				if (schemeCUDAIndex == SchemeCUDAEnum.VarDirCUDA) return true;
				if (schemeCUDAIndex == SchemeCUDAEnum.MultiGridCUDA) return true;
			}
			if (schemeCPUIndex == SchemeCPUEnum.SimpleIteration) return true;
			if (schemeCPUIndex == SchemeCPUEnum.SlidingIteration) return true;
			if (schemeCPUIndex == SchemeCPUEnum.SOR) return true;
			if (schemeCPUIndex == SchemeCPUEnum.Splitting) return true;
			if (schemeCPUIndex == SchemeCPUEnum.VarDir) return true;
			if (schemeCPUIndex == SchemeCPUEnum.PTM) return true;
			if (schemeCPUIndex == SchemeCPUEnum.GradientDescent) return true;
			if (schemeCPUIndex == SchemeCPUEnum.MinimumResidual) return true;
			if (schemeCPUIndex == SchemeCPUEnum.ConjugateGradient) return true;
			if (schemeCPUIndex == SchemeCPUEnum.BiconjugateGradient) return true;
			if (schemeCPUIndex == SchemeCPUEnum.Chebishev3Layers) return true;
			if (schemeCPUIndex == SchemeCPUEnum.MultiGrid) return true;

			return false;
		}

		string inputDataDirectory()
		{
			return Path.Combine(Utils.getDataPath(), "InputData");
		}

		void readInputData()
		{
			InputParser inputParser = new InputParser();

			try
			{
				var txtFiles = Directory.EnumerateFiles(inputDataDirectory(), "*.txt", SearchOption.AllDirectories);
				foreach (string currentFile in txtFiles)
				{
					VLPRectangleParams curParams = inputParser.parseFile(currentFile);
					if (curParams != null) listParams.Add(curParams);
				}
			}
			catch (Exception e)
			{
				MessageBox.Show(e.Message, Resources.strReadInputData);
			}
			if (listParams.Count == 0)
			{
				showCalculate = false;
				MessageBox.Show(string.Format(Resources.strNoInputData, inputDataDirectory()), "", MessageBoxButton.OK, MessageBoxImage.Information);
			}
		}

	}

	public interface IVLPRectangleInput
	{
		VLPRectangleParams getInputParameters();
		bool shouldVisualize();
		bool shouldSaveAnimatedGIF();
		InterpolationEnumVLPRectangle getInterpolationIndex();
		int precisionIndex();
		PlatformAndSchemeIndex getPlatformAndSchemeIndex();
		bool shouldMultiThread();
		bool shouldUseChebysh();
		bool shouldUseJordan();
		bool shouldUseBiconjugateStabilized();
		bool shouldUseVarSepProgonka();
		VarSepMethodsEnum varSepMethod();
		CRMethodsEnum crMethod();
		int getLParam();
		PlatformOCL oclPlatform();
		DeviceOCL oclDevice();
		int cudaDevice();
		bool shouldUseCompareAnalytic();
		int maxIterations();
		int getVisualStep();
		bool methodIsDirectOrFixedIterations();
		bool isOpenCLUsing();
		bool isCUDAUsing();
		int deviceOCLIndex();
		int deviceCUDAIndex();
		void setOutput(IVLPRectangleOutput inValue);
	}
}

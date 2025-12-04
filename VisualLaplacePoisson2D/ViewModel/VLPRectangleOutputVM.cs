using DD128Numeric;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using VLP2D.Model;
using VLP2D.Properties;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.ViewModel
{
	public class VLPRectangleOutputVM : ObservableObject, IVLPRectangleOutput
	{
		IVLPRectangleModel pModel = null;
		IVLPRectangleInput input = null;
		List<BitmapSource> listMap = new List<BitmapSource>();
		List<BitmapSource> listMapDiff = new List<BitmapSource>();
		bool bPrepareCalculationWasCalled;

		public void setInput(IVLPRectangleInput inValue)
		{
			input = inValue;
		}

		BitmapSource createHeatMapDiff(MinMaxF minMax, Adapter2D<float> adapter)
		{
			BitmapSource rc = UtilsPict.createHeatMap(palNoPurple, minMax, adapter);
			heatMapDiff = rc;
			return rc;
		}

		BitmapSource createInterpolatedHeatMap(bool palWithTransparent, MinMaxF minMax, Adapter2D<float> adapter, float stepX, float stepY, int width, int height)
		{
			BitmapSource rc = UtilsPict.createInterpolatedHeatMap(palWithTransparent ? Utils.palNoPurpleWithTransparent : Utils.palNoPurple, minMax, adapter, stepX, stepY, width, height);
			heatMap = rc;
			return rc;
		}

		Tuple<int, int> calculateWidthHeight(int cXSegments, double stepX, int cYSegments, double stepY)
		{
			if (cXSegments < UtilsPict.pictDim && cYSegments < UtilsPict.pictDim)
			{
				return new Tuple<int, int>(cXSegments + 1, cYSegments + 1);
			}
			int width = 0, height = UtilsPict.pictDim;
			double lngX = cXSegments * stepX;
			double lngY = cYSegments * stepY;
			double ratio = lngX / lngY;
			width = (int)(ratio * height);
			return new Tuple<int, int>(width, height);
		}

		public void doPrepareCalculation()
		{
			bPrepareCalculationWasCalled = true;

			VLPRectangleParams inputValues = input.getInputParameters();
			if (inputValues == null) return;

			InterpolationEnumVLPRectangle idxInterpol = input.getInterpolationIndex();
			PlatformAndSchemeIndex platformScheme = input.getPlatformAndSchemeIndex();
			bool isMultiThread = input.shouldMultiThread();

			MethodsParams miscParams;
			miscParams.isJordan = input.shouldUseJordan();
			miscParams.isChebysh = input.shouldUseChebysh();
			miscParams.isBiconjugateStabilized = input.shouldUseBiconjugateStabilized();
			miscParams.isVarSepProgonka = input.shouldUseVarSepProgonka();
			miscParams.methodVarSep = input.varSepMethod();
			miscParams.methodCR = input.crMethod();
			miscParams.isCompareAnalytic = input.shouldUseCompareAnalytic();
			miscParams.paramL = input.getLParam();

			miscParams.platform = platformScheme.platrofm == PlatformEnum.OCL ? input.oclPlatform() : null;
			miscParams.device = platformScheme.platrofm == PlatformEnum.OCL ? input.oclDevice() : null;

			miscParams.cudaDevice = input.cudaDevice();

			pModel.recalculateSteps(inputValues, platformScheme, miscParams.isVarSepProgonka, miscParams.methodVarSep);

			showDeviation = input.shouldUseCompareAnalytic();
			idxHeatMap = 0;
			sliderTicks = 1.0;
			allIters = "";
			allTime = "";
			deviation = "";
			elapsedInfo = "";
			listMap.Clear();
			listMapDiff.Clear();
			progressValue = 0;

			var widthHeight = calculateWidthHeight(inputValues.cXSegments, inputValues.stepX, inputValues.cYSegments, inputValues.stepY);
			BitmapSource funcIHM(bool palWithTransparent, MinMaxF minMax, Adapter2D<float> adapter) => createInterpolatedHeatMap(palWithTransparent, minMax, adapter, (float)inputValues.stepX, (float)inputValues.stepY, widthHeight.Item1, widthHeight.Item2);

			setModelMethodsParams();
			changeModelMultiThread(isMultiThread);
			changeModelVisualParams(input.shouldVisualize() ,input.getVisualStep());
			pModel.prepareCalculation(inputValues, idxInterpol, platformScheme, listMap, listMapDiff, funcIHM, createHeatMapDiff);

			if (listMap.Count > 0) heatMap = listMap[0];
			heatMapDiff = (listMapDiff.Count > 0) ? listMapDiff[0] : null;
		}

		public void setModelMethodsParams()
		{
			VLPRectangleParams inputValues = input.getInputParameters();
			if (inputValues == null) return;

			MethodsParams miscParams;
			miscParams.isJordan = input.shouldUseJordan();
			miscParams.isChebysh = input.shouldUseChebysh();
			miscParams.isBiconjugateStabilized = input.shouldUseBiconjugateStabilized();
			miscParams.isVarSepProgonka = input.shouldUseVarSepProgonka();
			miscParams.methodVarSep = input.varSepMethod();
			miscParams.methodCR = input.crMethod();
			miscParams.isCompareAnalytic = input.shouldUseCompareAnalytic();
			miscParams.paramL = input.getLParam();

			miscParams.platform = input.oclPlatform();
			miscParams.device = input.oclDevice();

			miscParams.cudaDevice = input.cudaDevice();

			pModel.setMethodParams(miscParams);
		}

		public void changeModelMultiThread(bool isMultiThread)
		{
			pModel.changeMultiThread(isMultiThread);
		}

		public void changeModelVisualParams(bool visualize, int stepHeatMap)
		{
			pModel.changeVisualParams(visualize, stepHeatMap);
		}

		public void allIterations()
		{
			string strErrors = pModel.stringFunctionExpressionErrors();
			if (strErrors != null)
			{
				MessageBox.Show(strErrors, Resources.strExpressionError);
				return;
			}
			bShowProgress = true;
			if (!bPrepareCalculationWasCalled) doPrepareCalculation();
			bPrepareCalculationWasCalled = false;

			saveTaskParams();

			pModel.allIterations();
		}

		void saveTaskParams()
		{
			TaskParams taskParams = new TaskParams();

			taskParams.isOpenCLCheckBoxChecked = input.isOpenCLUsing();
			taskParams.isCUDACheckBoxChecked = input.isCUDAUsing();
			taskParams.idxScheme = input.getPlatformAndSchemeIndex().idxScheme;
			taskParams.precision = input.precisionIndex();
			taskParams.idxInterpol = (int)input.getInterpolationIndex();
			taskParams.idxDeviceOCL = input.deviceOCLIndex();
			taskParams.idxDeviceCUDA = input.deviceCUDAIndex();

			taskParams.isJordan = input.shouldUseJordan();
			taskParams.isChebysh = input.shouldUseChebysh();
			taskParams.isBiconjugateStabilized = input.shouldUseBiconjugateStabilized();
			taskParams.isVarSepProgonka = input.shouldUseVarSepProgonka();
			taskParams.varSepMethod = input.varSepMethod();
			taskParams.crMethod = input.crMethod();
			taskParams.paramL = input.getLParam();

			taskParams.save(input.getInputParameters().name);
		}

		void progressEventHandler(double percent)
		{
			progressValue = percent;
		}

		void progressHeaderEventHandler(string header)
		{
			progressHeader = header;
		}

		void completedEventHandler()
		{
			bShowProgress = false;

			sliderTicks = listMap.Count - 1;
			sliderLargeChange = sliderTicks / 20.0;
			if (sliderLargeChange < 1.0) sliderLargeChange = 1;
			IterationsKind itersKind = pModel.getIterationsKind();
			showIterations = itersKind != IterationsKind.None;
			string strIters = "";
			if (itersKind != IterationsKind.None)
			{
				strIters = String.Format("{0}", pModel.getElapsedIters());
				if (itersKind == IterationsKind.unknown) strIters += String.Format("({0})", pModel.getAllIters());
			}
			allIters = strIters;
			if (pModel.getInitTime() > 0.1) allTime = String.Format(CultureInfo.InvariantCulture, "{0:0.0##} {1} {2:0.0##}", pModel.getElapsedTime(), Resources.strInitialization, pModel.getInitTime());
			else allTime = String.Format(CultureInfo.InvariantCulture, "{0:0.0##}", pModel.getElapsedTime());
			if (showDeviation) deviation = pModel.getDeviation();
			elapsedInfo = pModel.getElapsedInfo();

			if (input.shouldSaveAnimatedGIF()) saveAnimatedGIF();
		}

		void saveAnimatedGIF()
		{
			string dstDirectory = Path.Combine(Utils.getDataPath(), "OutputData");
			Directory.CreateDirectory(dstDirectory);

			PlatformAndSchemeIndex platformScheme = input.getPlatformAndSchemeIndex();
			Type enumType = typeof(SchemeCPUEnum);
			switch (platformScheme.platrofm)
			{
				case PlatformEnum.CPU:
					enumType = typeof(SchemeCPUEnum);
					break;
				case PlatformEnum.OCL:
					enumType = typeof(SchemeOCLEnum);
					break;
				case PlatformEnum.CUDA:
					enumType = typeof(SchemeCUDAEnum);
					break;
			}
			string methodName = Enum.GetName(enumType, platformScheme.idxScheme);
			string outputFileName = input.getInputParameters().name + "_" + methodName + ".gif";

			FilmStrip film = new FilmStrip();
			film.Start();
			for (int i = 1; i < listMap.Count; i++) film.AddFrame(listMap[i]);//skip 1st map, which is different size, has no transparency
			film.SaveToFile(Path.Combine(dstDirectory, outputFileName));
		}

		public ICommand allIterationsCancelCommand
		{
			get { return new DelegateCommand(() => pModel.cancelAll()); }
		}

		double _sliderTicks = 0;
		public double sliderTicks
		{
			get { return _sliderTicks; }
			set
			{
				_sliderTicks = value;
				RaisePropertyChangedEvent("sliderTicks");
			}
		}

		int _idxHeatMap = 0;
		public int idxHeatMap
		{
			get { return _idxHeatMap; }
			set
			{
				if (value >= listMap.Count) return;
				if (_idxHeatMap != value)
				{
					_idxHeatMap = value;

					heatMap = listMap[_idxHeatMap];
					if (_idxHeatMap < listMapDiff.Count) heatMapDiff = listMapDiff[_idxHeatMap];
				}
			}
		}

		double _sliderLargeChange = 10;
		public double sliderLargeChange
		{
			get { return _sliderLargeChange; }
			set
			{
				_sliderLargeChange = value;
				RaisePropertyChangedEvent("sliderLargeChange");
			}
		}

		string _allIters = "";
		public string allIters
		{
			get { return _allIters; }
			set
			{
				_allIters = value;
				RaisePropertyChangedEvent("allIters");
			}
		}

		bool _showIterations = true;
		public bool showIterations
		{
			get { return _showIterations; }
			set
			{
				if (_showIterations == value) return;
				_showIterations = value;
				RaisePropertyChangedEvent("showIterations");
			}
		}

		string _allTime = "";
		public string allTime
		{
			get { return _allTime; }
			set
			{
				_allTime = value;
				RaisePropertyChangedEvent("allTime");
			}
		}

		string _deviation = "";
		public string deviation
		{
			get { return _deviation; }
			set
			{
				_deviation = value;
				RaisePropertyChangedEvent("deviation");
			}
		}

		string _elapsedInfo = "";
		public string elapsedInfo
		{
			get { return _elapsedInfo; }
			set
			{
				_elapsedInfo = value;
				RaisePropertyChangedEvent("elapsedInfo");
			}
		}

		bool _showDeviation = false;
		public bool showDeviation
		{
			get { return _showDeviation; }
			set
			{
				if (_showDeviation == value) return;
				_showDeviation = value;
				RaisePropertyChangedEvent("showDeviation");
			}
		}

		public bool bShowSlider
		{
			get { return !bShowProgress; }
		}

		bool _bShowProgress = false;
		public bool bShowProgress
		{
			get { return _bShowProgress; }
			set
			{
				if (_bShowProgress == value) return;
				_bShowProgress = value;
				RaisePropertyChangedEvent("bShowProgress");
				RaisePropertyChangedEvent("bShowSlider");
			}
		}

		double _progressValue = 0;
		public double progressValue
		{
			get { return _progressValue; }
			set
			{
				_progressValue = value;
				RaisePropertyChangedEvent("progressValue");
			}
		}

		string _progressHeader = "";
		public string progressHeader
		{
			get { return _progressHeader; }
			set
			{
				_progressHeader = value;
				RaisePropertyChangedEvent("progressHeader");
			}
		}

		ImageSource _heatMap;
		public ImageSource heatMap
		{
			get
			{
				return _heatMap;
			}
			set
			{
				_heatMap = value;
				RaisePropertyChangedEvent("heatMap");
			}
		}

		ImageSource _heatMapDiff;
		public ImageSource heatMapDiff
		{
			get
			{
				return _heatMapDiff;
			}
			set
			{
				_heatMapDiff = value;
				RaisePropertyChangedEvent("heatMapDiff");
			}
		}

		public VLPRectangleOutputVM()
		{
		}

		public void changeModelPrecision(int indexPrecision)
		{
			pModel?.removeProgressHandler(progressEventHandler);
			pModel?.removeProgressHeaderHandler(progressHeaderEventHandler);
			pModel?.removeCompletedHandler(completedEventHandler);

			switch (indexPrecision)
			{
				case 0:
					pModel = new VLPRectangleModel<float>();
					break;
				case 1:
					pModel = new VLPRectangleModel<double>();
					break;
				case 2:
					pModel = new VLPRectangleModel<DD128>();
					break;
				case 3:
					pModel = new VLPRectangleModel<QD256>();
					break;
			}

			pModel?.addProgressHandler(progressEventHandler);
			pModel?.addProgressHeaderHandler(progressHeaderEventHandler);
			pModel?.addCompletedHandler(completedEventHandler);
		}

		public int getModelMaxIterations(PlatformAndSchemeIndex platformScheme, VLPRectangleParams pParams, InterpolationEnumVLPRectangle idxInterpolation)
		{
			if (input.methodIsDirectOrFixedIterations()) return 1;
			return pModel.getMaxIterations(platformScheme, pParams, idxInterpolation);
		}

		public void setModelIterations(int iters)
		{
			pModel.setMaxIterations(iters);
		}

		public void compileModelFunctions(VLPRectangleParams pParams)
		{
			pModel.compileFunctions(pParams);
		}
	}

	public interface IVLPRectangleOutput
	{
		void doPrepareCalculation();
		void setModelMethodsParams();
		void changeModelMultiThread(bool isMultiThread);
		void changeModelVisualParams(bool visualize, int stepHeatMap);
		void changeModelPrecision(int indexPrecision);
		int getModelMaxIterations(PlatformAndSchemeIndex platformScheme, VLPRectangleParams pParams, InterpolationEnumVLPRectangle idxInterpolation);
		void setModelIterations(int iters);
		void compileModelFunctions(VLPRectangleParams pParams);
		void allIterations();
		void setInput(IVLPRectangleInput inValue);
	}
}

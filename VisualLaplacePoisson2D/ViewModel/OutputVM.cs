using DD128Numeric;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using VLP2D.Common;
using VLP2D.Model;
using VLP2D.Properties;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.ViewModel
{
	public class OutputVM : ObservableObject, IOutputVM
	{
		IModel pModel = null;
		IInputVM input = null;
		List<BitmapSource> listMap = new List<BitmapSource>();
		List<BitmapSource> listMapDiff = new List<BitmapSource>();
		bool bPrepareCalculationWasCalled;
		float[,] interpolatedArray;
		Adapter2D<float> adapterInterpolatedArray;
		Picture3D picture3D;

		public void setInput(IInputVM inValue)
		{
			input = inValue;
		}

		BitmapSource createHeatMapDiff(MinMaxF minMax, Adapter2D<float> adapter, int width, int height)
		{
			scaleYDif = ((double)height / (double)width) * ((double)adapter.dim1 / (double)adapter.dim2);
			BitmapSource rc = UtilsPict.createHeatMap(palNoPurple, minMax, adapter);
			heatMapDiff = rc;
			return rc;
		}

		BitmapSource createInterpolatedHeatMap(bool palWithTransparent, MinMaxF minMax, Adapter2D<float> adapter, float stepX, float stepY, int width, int height)
		{
			BitmapSource rc;
			BitmapPalette pal = palWithTransparent ? Utils.palNoPurpleWithTransparent : Utils.palNoPurple;

			float fMin, fMax;
			interpolatedArray = createInterpolatedArray(adapter, stepX, stepY, width, height, out fMin, out fMax);
			adapterInterpolatedArray = new Adapter2D<float>(width, height, (i, j) => interpolatedArray[i, j]);
			rc = createHeatMap(pal, new MinMaxF(fMin, fMax), adapterInterpolatedArray, 0);

			heatMap = rc;
			return rc;
		}

		void createSurface(Adapter2D<float> adapter, float xMin, float xMax, float yMin, float yMax)
		{
			UtilsThread.runOnUIThread(() =>
			{
				modelSurface = picture3D.plot(adapter, new MinMaxF(xMin, xMax), new MinMaxF(yMin, yMax));
			});
		}

		Tuple<int, int> calculateWidthHeight(int cXSegments, double stepX, int cYSegments, double stepY)
		{
			double lngX = cXSegments * stepX;
			double lngY = cYSegments * stepY;
			int width = 0, height = 0;
			if (lngX >= lngY)
			{
				width = UtilsPict.pictDim;
				double ratio = lngX / lngY;
				height = (int)(width / ratio);
			}
			else
			{
				height = UtilsPict.pictDim;
				double ratio = lngY / lngX;
				width = (int)(height / ratio);
			}
			return new Tuple<int, int>(width, height);
		}

		public void doPrepareCalculation()
		{
			bPrepareCalculationWasCalled = true;

			TaskInputParams inputValues = input.getInputParameters();
			if (inputValues == null) return;

			InterpolationEnum idxInterpol = input.getInterpolationIndex();
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

			RectangleDataDouble rdd = pModel.recalculateSteps(inputValues, platformScheme, miscParams.isVarSepProgonka, miscParams.methodVarSep);

			showDeviation = input.shouldUseCompareAnalytic();
			idxHeatMap = 0;
			sliderTicks = 1.0;
			allIters = "";
			allTime = "";
			deviation = "";
			elapsedInfo = "";
			listMap.Clear();
			listMapDiff.Clear();
			modelSurface = null;
			picture3D.reset();
			progressValue = 0;

			var widthHeight = calculateWidthHeight(rdd.cXSegments, rdd.stepX, rdd.cYSegments, rdd.stepY);
			BitmapSource funcIHM(bool palWithTransparent, MinMaxF minMax, Adapter2D<float> adapter) => createInterpolatedHeatMap(palWithTransparent, minMax, adapter, (float)rdd.stepX, (float)rdd.stepY, widthHeight.Item1, widthHeight.Item2);
			BitmapSource funcDiff(MinMaxF minMax, Adapter2D<float> adapter) => createHeatMapDiff(minMax, adapter, widthHeight.Item1, widthHeight.Item2);

			void funcCreateSurface() => createSurface(adapterInterpolatedArray, (float)rdd.xMin, (float)rdd.xMax, (float)rdd.yMin, (float)rdd.yMax);

			setModelMethodsParams();
			changeModelMultiThread(isMultiThread);
			changeModelVisualParams(input.shouldVisualize() ,input.getVisualStep());
			pModel.prepareCalculation(inputValues, idxInterpol, platformScheme, listMap, listMapDiff, funcIHM, funcDiff, funcCreateSurface);

			if (listMap.Count > 0) heatMap = listMap[0];
			heatMapDiff = (listMapDiff.Count > 0) ? listMapDiff[0] : null;
		}

		public void setModelMethodsParams()
		{
			TaskInputParams inputValues = input.getInputParameters();
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
			TaskCalcParams taskParams = new TaskCalcParams();

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

		public ICommand switchMode2D3DCommand
		{
			get { return new DelegateCommand(() => switchMode2D3D()); }
		}

		void switchMode2D3D()
		{
			bShow3D = !bShow3D;
			strButton2D3D = bShow3D ? "3D -> 2D" : "2D -> 3D";
		}

		public ICommand mouseLeftButtonDown
		{
			get { return new ParameterCommand((arg) => doMouseLeftButtonDown((MouseButtonEventArgs)arg)); }
		}

		public ICommand mouseMove
		{
			get { return new ParameterCommand((arg) => doMouseMove((MouseEventArgs)arg)); }
		}

		public ICommand mouseLeftButtonUp
		{
			get { return new ParameterCommand((arg) => doMouseLeftButtonUp((MouseButtonEventArgs)arg)); }
		}

		public ICommand homeCommand
		{
			get { return new DelegateCommand(() => doHome()); }
		}

		public ICommand zoomInCommand
		{
			get { return new DelegateCommand(() => doZoomIn()); }
		}

		public ICommand zoomOutCommand
		{
			get { return new DelegateCommand(() => doZoomOut()); }
		}

		void doMouseLeftButtonDown(MouseButtonEventArgs arg)
		{
			if (bShow3D && arg.LeftButton == MouseButtonState.Pressed)
			{
				picture3D.mouseDown(arg.GetPosition(null));
			}
		}

		void doMouseMove(MouseEventArgs arg)
		{
			if (bShow3D && arg.LeftButton == MouseButtonState.Pressed)
			{
				picture3D.mouseMove(arg.GetPosition(null), gridWidth, gridHeight);
			}
		}

		void doMouseLeftButtonUp(MouseButtonEventArgs arg)
		{
			if (bShow3D) picture3D.mouseUp();
		}

		void doHome()
		{
			if (bShow3D) picture3D.home();
		}

		void doZoomIn()
		{
			if (bShow3D) picture3D.zoomIn();
		}

		void doZoomOut()
		{
			if (bShow3D) picture3D.zoomOut();
		}

		bool _bShow3D = false;
		public bool bShow3D
		{
			get { return _bShow3D; }
			set
			{
				if (_bShow3D == value) return;
				_bShow3D = value;
				RaisePropertyChangedEvent("bShow3D");
			}
		}

		string _strButton2D3D = "2D -> 3D";
		public string strButton2D3D
		{
			get { return _strButton2D3D; }
			set
			{
				if (_strButton2D3D == value) return;
				_strButton2D3D = value;
				RaisePropertyChangedEvent("strButton2D3D");
			}
		}

		Model3D _modelSurface = null;
		public Model3D modelSurface
		{
			get { return _modelSurface; }
			set
			{
				if (_modelSurface == value) return;
				_modelSurface = value;
				RaisePropertyChangedEvent(nameof(modelSurface));
			}
		}

		private double _gridHeight;
		public double gridHeight
		{
			get { return _gridHeight; }
			set
			{
				if (value == _gridHeight) return;
				_gridHeight = value;
				RaisePropertyChangedEvent(nameof(gridHeight));
			}
		}

		private double _gridWidth;
		public double gridWidth
		{
			get { return _gridWidth; }
			set
			{
				if (value == _gridWidth) return;
				_gridWidth = value;
				RaisePropertyChangedEvent(nameof(gridWidth));
			}
		}

		private double _scaleYDif = 1.0;
		public double scaleYDif
		{
			get { return _scaleYDif; }
			set
			{
				if (value == _scaleYDif) return;
				_scaleYDif = value;
				RaisePropertyChangedEvent(nameof(scaleYDif));
			}
		}

		public OutputVM()
		{
			picture3D = new Picture3D();
		}

		public void changeModelPrecision(int indexPrecision)
		{
			pModel?.removeProgressHandler(progressEventHandler);
			pModel?.removeProgressHeaderHandler(progressHeaderEventHandler);
			pModel?.removeCompletedHandler(completedEventHandler);

			switch (indexPrecision)
			{
				case 0:
					pModel = new Model<float>();
					break;
				case 1:
					pModel = new Model<double>();
					break;
				case 2:
					pModel = new Model<DD128>();
					break;
				case 3:
					pModel = new Model<QD256>();
					break;
			}

			pModel?.addProgressHandler(progressEventHandler);
			pModel?.addProgressHeaderHandler(progressHeaderEventHandler);
			pModel?.addCompletedHandler(completedEventHandler);
		}

		public int getModelMaxIterations(PlatformAndSchemeIndex platformScheme, TaskInputParams pParams, InterpolationEnum idxInterpolation)
		{
			if (input.methodIsDirectOrFixedIterations()) return 1;
			return pModel.getMaxIterations(platformScheme, pParams, idxInterpolation);
		}

		public void setModelIterations(int iters)
		{
			pModel.setMaxIterations(iters);
		}

		public void compileModelFunctions(TaskInputParams pParams)
		{
			pModel.compileFunctions(pParams);
		}
	}

	public interface IOutputVM
	{
		void doPrepareCalculation();
		void setModelMethodsParams();
		void changeModelMultiThread(bool isMultiThread);
		void changeModelVisualParams(bool visualize, int stepHeatMap);
		void changeModelPrecision(int indexPrecision);
		int getModelMaxIterations(PlatformAndSchemeIndex platformScheme, TaskInputParams pParams, InterpolationEnum idxInterpolation);
		void setModelIterations(int iters);
		void compileModelFunctions(TaskInputParams pParams);
		void allIterations();
		void setInput(IInputVM inValue);
	}
}

using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	internal class MarchingSchemeCU<T> : Direct1DSeparateBoundariesScheme<T>, IScheme<T> where T : struct, INumber<T>, IMinMaxValue<T>, ITrigonometricFunctions<T>, IRootFunctions<T>
	{
		CudaContext ctx;
		CudaDeviceVariable<T> pq0, pq1, pq2;
		CudaDeviceVariable<T> fnCU, unCU;
		CudaKernel kernelCalculatePkQk, kernelPrepareFFT, kernelReducedSystem, kernelPlaceWToUn, kernelCalculateY;

		T[] fn;
		T[] unShow;
		Func<T, T, T> fKsi;
		readonly int k, L, L2, M, cCores, fftInOutSize;
		T stepX, stepY, cBase, ai, bi, subsupra, normFactor;//subsupra - subdiagonal,supradiagonal
		List<BitmapSource> lstBitmap;

		Action<double> reportProgress;
		float curProgress, oldProgress;
		bool iterationsCanceled;
		CancellationTokenSource cts = new CancellationTokenSource();
		readonly ParallelOptions optionsParallel;
		int dim1Fn, dim2Fn;
		IFFTCU<T> fft;
		Dictionary<int, CudaFFTPlanMany> plans;
		T _2 = T.CreateTruncating(2);
		T _Ny;

		public MarchingSchemeCU(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsiIn, int paramL, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn, int cudaDevice) :
			base(cXSegments, cYSegments, fCreateBitmap)
		{
			ctx = new CudaContext(cudaDevice);
			dim1Fn = Nx - 1;
			dim2Fn = Ny + 1;
			fn = new T[dim1Fn * dim2Fn];

			L = paramL;//max value == (Nx - 1) / 4, because k is minimum 2
			L2 = L * 2;
			k = (Nx - 1) / L2;//k > 1 always

			M = Ny - 1;
			_Ny = T.CreateTruncating(Ny);
			//Bank Rose Marching Algorithms for Elliptic Boundary Value Problems. I The Constant Coefficient Case.pdf p.21(6.8)
			normFactor = T.Sqrt(_2 / _Ny);//normalizing factor;Demmel rus 281

			stepX = stepXIn;
			stepY = stepYIn;
			subsupra = stepX * stepX / (stepY * stepY);//[SNR] p.106, (4)
			ai = subsupra;//subdiagonal,[SNR] p.106, (4)
			bi = subsupra;//supradiagonal,[SNR] p.106, (4)

			fKsi = fKsiIn;

			lstBitmap = lstBitmap0;
			if (lstBitmap != null) unShow = new T[(Nx + 1) * (Ny + 1)];

			reportProgress = reportProgressIn;
			curProgress = 0;
			oldProgress = 0;

			cBase = (T.One + subsupra) * _2;//[SNR] p.106, (4)

			this.cCores = cCores;
			optionsParallel = new ParallelOptions() { MaxDegreeOfParallelism = cCores, CancellationToken = cts.Token };

			if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
			{
				plans = new Dictionary<int, CudaFFTPlanMany>();
				fft = new FFTCU<T>(ctx, plans, Ny, true);
			}
			else fft = new FFTLomontCU<T>(ctx, Ny);

			fftInOutSize = (Ny / 2 + 1) * FFTConstant.sizeOfComplex;

			string functionNameCalculatePQVectors = "calculatePQVectors";
			string functionNamePrepareFFT = "prepareFFT";
			string functionNameSolveReducedSystem = "solveReducedSystem";
			string functionNamePlaceWToUn = "placeWToUn";
			string functionNameCalculateY = "calculateY";
			string moduleName = UtilsCU.moduleName("Marching_", Utils.getTypeName<T>(), ctx.DeviceId);
			CUmodule? module;

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string defines =
@"
#define overDiag(i, overUnder)	(((i & 1) == 0) ? overUnder : One)
#define underDiag(i, overUnder)	(((i & 1) == 1) ? overUnder : One)
";
				string constants =
@"
static __device__ __constant__ int L, k, Nx, M, colsPQ, colsFn, colsUn, upperL, alfaUB;
static __device__ __constant__ bool ksiIsNull;
";
				string strTypeName = Utils.getTypeName<T>();
				string strProgram = "";
				if (typeof(T) == typeof(float)) strProgram = HighPrecisionCU.strSingleDefines;
				if (typeof(T) == typeof(double)) strProgram = HighPrecisionCU.strDoubleDefines;
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + HighPrecisionCU.strDD128Trig;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + HighPrecisionCU.strQD256Trig;
				strProgram += defines + constants;

				strProgram += MarchingKernelCalculatePQVectorsCU.createProgramCalculatePQVectors(functionNameCalculatePQVectors, strTypeName);
				strProgram += MarchingKernelPrepareFFTCU.createProgramPrepareFFT(functionNamePrepareFFT, strTypeName);
				strProgram += MarchingKernelReducedSystemCU.createProgramReducedSystem(functionNameSolveReducedSystem, strTypeName);
				strProgram += MarchingKernelPlaceWToUnCU.createProgramPlaceWToUn(functionNamePlaceWToUn, strTypeName);
				strProgram += MarchingKernelCalculateYCU.createProgramCalculateY(functionNameCalculateY, strTypeName);

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}
			kernelCalculatePkQk = new CudaKernel(functionNameCalculatePQVectors, (CUmodule)module);
			kernelPrepareFFT = new CudaKernel(functionNamePrepareFFT, (CUmodule)module);
			kernelReducedSystem = new CudaKernel(functionNameSolveReducedSystem, (CUmodule)module);
			if (typeof(T) == typeof(DD128)) HighPrecisionCU.setTrogonometricConstantsDD128(kernelReducedSystem);
			if (typeof(T) == typeof(QD256)) HighPrecisionCU.setTrogonometricConstantsQD256(kernelReducedSystem);
			kernelPlaceWToUn = new CudaKernel(functionNamePlaceWToUn, (CUmodule)module);
			kernelCalculateY = new CudaKernel(functionNameCalculateY, (CUmodule)module);

			kernelCalculatePkQk.SetConstantVariable("L", L);
			kernelCalculatePkQk.SetConstantVariable("k", k);
			kernelCalculatePkQk.SetConstantVariable("Nx", Nx);
			kernelCalculatePkQk.SetConstantVariable("M", M);
			kernelCalculatePkQk.SetConstantVariable("colsPQ", fftInOutSize);
			kernelCalculatePkQk.SetConstantVariable("colsFn", dim2Fn);
			kernelCalculatePkQk.SetConstantVariable("colsUn", dim2Fn);
			kernelCalculatePkQk.SetConstantVariable("upperL", L * 2);
			kernelCalculatePkQk.SetConstantVariable("alfaUB", L - 1);
			kernelCalculatePkQk.SetConstantVariable("ksiIsNull", fKsi == null);
		}

		public T doIteration(int iter)
		{
			initElapsedList();
			float elapsed = getExecutedSeconds(stopWatchEL, () => initFj());
			listElapsedAdd("initFj", elapsed);

			try
			{
				forwardPath();
				if (iterationsCanceled) return T.One;

				solveReducedSystem();
				if (iterationsCanceled) return T.One;

				reversePath();
				if (iterationsCanceled) return T.One;

				un = unCU;//instead of un = new float[(Nx + 1) * (Ny + 1)] and unCU.CopyToHost(un)

				elapsed = getExecutedSeconds(stopWatchEL, () => restoreBounds(un));
				listElapsedAdd("restoreBounds", elapsed);
			}
			catch (Exception)
			{
				cleanup();
			}

			return T.One;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		void initFj()
		{
			T stepX2 = stepX * stepX;
			if (fKsi != null) GridIterator.iterate(0, Nx - 1, 1, Ny, (i, j) => { fn[i * dim2Fn + j] += stepX2 * fKsi(stepX * T.CreateTruncating(i + 1), stepY * T.CreateTruncating(j)); });//[SNR] p.123 (8),[SNE] p.120 (8)

			Parallel.For(0, Nx - 1, optionsParallel, (i) =>
			{
				fn[i * dim2Fn + 1] += subsupra * bndB[i];//[SNR] p.123 (8),[SNE] p.120 (8)
				fn[i * dim2Fn + Ny - 1] += subsupra * bndT[i];//[SNR] p.123 (8),[SNE] p.120 (8)
			});

			Parallel.For(1, Ny, optionsParallel, (j) =>
			{
				fn[0 * dim2Fn + j] += bndL[j - 1];//analog [SNR] p.123 (8),[SNE] p.120 (8)
				fn[(Nx - 2) * dim2Fn + j] += bndR[j - 1];//analog [SNR] p.123 (8),[SNE] p.120 (8)
			});

			fnCU = fn;
			fn = null;
		}

		void forwardPath()
		{
			calculatePQVectors();
			showProgress(20);
			if (iterationsCanceled) return;

			calculateĜ();
			showProgress(20);
		}

		void calculatePQVectors()
		{
			pq0 = new CudaDeviceVariable<T>(L2 * fftInOutSize);
			pq1 = new CudaDeviceVariable<T>(L2 * fftInOutSize);
			pq2 = new CudaDeviceVariable<T>(L2 * fftInOutSize);

			object[] argsCalculatePkQk = [pq0.DevicePointer, pq1.DevicePointer, pq2.DevicePointer, fnCU.DevicePointer, cBase, ai, bi];
			UtilsCU.set1DKernelDims(kernelCalculatePkQk, L2);

			float elapsed = kernelCalculatePkQk.Run(argsCalculatePkQk) / 1000.0f;
			listElapsedAdd("CalculatePkQk", elapsed);
		}

		void calculateĜ()
		{
			swapPQArrays();

			object[] argsPrepareFFT = [pq0.DevicePointer, pq1.DevicePointer, pq2.DevicePointer];
			UtilsCU.set2DKernelDims(kernelPrepareFFT, L2, M);

			float elapsed = kernelPrepareFFT.Run(argsPrepareFFT) / 1000.0f;
			listElapsedAdd("PrepareFFT", elapsed);

			UtilsCU.disposeBuf(ref pq1);
			UtilsCU.disposeBuf(ref pq2);

			elapsed = getExecutedSeconds(stopWatchEL, () => fft.calculate(pq0, L2, normFactor));
			listElapsedAdd("fft 1", elapsed);
		}

		void swapPQArrays()
		{
			int rem = (k - 1) % 3;
			if (rem == 1)
			{//tmp <- 0 <- 1 <- 2 <- tmp
				CudaDeviceVariable<T> tmp = pq0;
				pq0 = pq1;
				pq1 = pq2;
				pq2 = tmp;
			}
			else if (rem == 2)
			{//tmp <- 2 <- 1 <- 0 <- tmp
				CudaDeviceVariable<T> tmp = pq2;
				pq2 = pq1;
				pq1 = pq0;
				pq0 = tmp;
			}
		}

		void solveReducedSystem()
		{
			CudaDeviceVariable<T> φ = null, α = null;
			try
			{
				φ = new CudaDeviceVariable<T>(M * L2);
				α = new CudaDeviceVariable<T>(M * L);

				object[] argsReducedSystem = [φ.DevicePointer, α.DevicePointer, pq0.DevicePointer, T.Pi / (_2 * _Ny), subsupra * _2];//pq0 as ŵ
				UtilsCU.set1DKernelDims(kernelReducedSystem, M);

				float elapsed2 = kernelReducedSystem.Run(argsReducedSystem) / 1000.0f;
				listElapsedAdd("ReducedSystem", elapsed2);
				showProgress(20);
			}
			finally
			{
				UtilsCU.disposeBuf(ref φ);
				UtilsCU.disposeBuf(ref α);
			}
		}

		void reversePath()
		{
			float elapsed = 0;

			elapsed = getExecutedSeconds(stopWatchEL, () => fft.calculate(pq0, L2, normFactor));//pq0 as ŵ
			listElapsedAdd("fft 2", elapsed);
			fft.cleanup();
			fft = null;
			showProgress(20);
			if (iterationsCanceled) return;

			try
			{
				unCU = new CudaDeviceVariable<T>((Nx + 1) * (Ny + 1));

				object[] argsPlaceWToUn = [pq0.DevicePointer, unCU.DevicePointer];
				UtilsCU.set1DKernelDims(kernelPlaceWToUn, L2);

				elapsed = kernelPlaceWToUn.Run(argsPlaceWToUn) / 1000.0f;
				listElapsedAdd("PlaceWToUn", elapsed);

				if (unShow != null)
				{
					for (int iW = 0; iW < 2 * L; iW++)
					{
						long iUN = (long)(iW * k + ((iW & 1) != 1 ? 1 : k));
						long offsetSrcDstInBytes = (iUN * (Ny + 1) + 1) * Marshal.SizeOf(typeof(T));
						unCU.CopyToHost(unShow, offsetSrcDstInBytes, offsetSrcDstInBytes, (Ny - 1) * Marshal.SizeOf(typeof(T)));
					}
					UtilsPict.addPicture(lstBitmap, true, null/*minMax*/, new Adapter2D<float>(Nx + 1, Ny + 1, (m, k) => float.CreateTruncating(unShow[m * (Ny + 1) + k])), fCreateBitmap);
				}
			}
			finally
			{
				UtilsCU.disposeBuf(ref pq0);
			}

			calculateY();
			showProgress(20);
		}

		void calculateY()
		{
			object[] argsCalculateY = [unCU.DevicePointer, fnCU.DevicePointer, cBase, ai, bi];
			UtilsCU.set1DKernelDims(kernelCalculateY, L2);

			float elapsed = kernelCalculateY.Run(argsCalculateY) / 1000.0f;
			listElapsedAdd("CalculateY", elapsed);
		}

		void restoreBounds(T[] uu)
		{
			int cols = Ny + 1;
			Parallel.For(1, Nx, optionsParallel, (i) =>
			{
				uu[i * cols + 0] = bndB[i - 1];
				uu[i * cols + Ny] = bndT[i - 1];
			});

			Parallel.For(1, Ny, optionsParallel, (j) =>
			{
				uu[0 * cols + j] = bndL[j - 1];
				uu[Nx * cols + j] = bndR[j - 1];
			});

			T nan = T.Zero / T.Zero;
			uu[0 * cols + 0] = nan;
			uu[Nx * cols + 0] = nan;
			uu[0 * cols + Ny] = nan;
			uu[Nx * cols + Ny] = nan;
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
			if (unShow != null)
			{
				GridIterator.iterateEdgesAndFillInternalPoints(Nx + 1, Ny + 1, null, (i, j) => unShow[i * (Ny + 1) + j] = T.Zero / T.Zero);//can't use IFloatingPointIeee754
				fillBounds(unShow);

				void fillBounds(T[] uu)
				{
					Parallel.For(1, Nx, optionsParallel, (i) =>
					{
						uu[i * (Ny + 1) + 0] = bndB[i - 1];
						uu[i * (Ny + 1) + Ny] = bndT[i - 1];
					});

					Parallel.For(1, Ny, optionsParallel, (j) =>
					{
						uu[0 * (Ny + 1) + j] = bndL[j - 1];
						uu[Nx * (Ny + 1) + j] = bndR[j - 1];
					});

					uu[0 * (Ny + 1) + 0] = (uu[1 * (Ny + 1) + 0] + uu[0 * (Ny + 1) + 1]) / _2;
					uu[Nx * (Ny + 1) + 0] = (uu[(Nx - 1) * (Ny + 1) + 0] + uu[Nx * (Ny + 1) + 1]) / _2;
					uu[0 * (Ny + 1) + Ny] = (uu[1 * (Ny + 1) + Ny] + uu[0 * (Ny + 1) + Ny - 1]) / _2;
					uu[Nx * (Ny + 1) + Ny] = (uu[(Nx - 1) * (Ny + 1) + Ny] + uu[Nx * (Ny + 1) + Ny - 1]) / _2;
				}
			}
		}

		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public virtual void cleanup()
		{
			UtilsCU.disposeBuf(ref unCU);
			UtilsCU.disposeBuf(ref fnCU);

			if (plans != null) foreach (var plan in plans) plan.Value.Dispose();
			plans = null;
			fft?.cleanup();
			fft = null;
			un = null;
			if (kernelCalculatePkQk != null) ctx?.UnloadModule(kernelCalculatePkQk.CUModule);
			ctx?.Dispose();
			ctx = null;
		}

		bool areIterationsCanceled()
		{
			return iterationsCanceled;
		}

		void showProgress(float count)
		{
			curProgress += count;
			if (curProgress - oldProgress > 0.99)
			{
				reportProgress((int)curProgress);
				oldProgress = curProgress;
			}
		}

		override public string getElapsedInfo() { return timesElapsed(); }
	}
}

using CLMathLibraries.CLFFT;
using Cloo;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	internal class MarchingSchemeOCL<T> : Direct1DSeparateBoundariesScheme<T>, IScheme<T> where T : struct, INumber<T>, IMinMaxValue<T>, ITrigonometricFunctions<T>, IRootFunctions<T>
	{
		CommandQueueOCL commands;
		BufferOCL<T> pq0, pq1, pq2;
		BufferOCL<T> fnOCL, unOCL;

		T[] fn;
		T[,] unShow;
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
		IFFTOCL<T> fft;
		Dictionary<int, CLFFTPlan> plans;
		T _2 = T.CreateTruncating(2);

		public MarchingSchemeOCL(int cXSegments, int cYSegments, T stepXIn, T stepYIn, int cCores, Func<T, T, T> fKsiIn, int paramL, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, PlatformOCL platform, DeviceOCL device, Action<double> reportProgressIn) :
			base(cXSegments, cYSegments, fCreateBitmap)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			commands = UtilsCL.createCommandQueue(platform, device, CommandQueueFlagsOCL.None);
			dim1Fn = Nx - 1;
			dim2Fn = Ny + 1;

			L = paramL;//max value == (Nx - 1) / 4, because k is minimum 2
			L2 = L * 2;
			k = (Nx - 1) / L2;//k > 1 always

			M = Ny - 1;

			//Bank Rose Marching Algorithms for Elliptic Boundary Value Problems. I The Constant Coefficient Case.pdf p.21(6.8)
			normFactor = T.Sqrt(T.CreateTruncating(2.0 / Ny));//normalizing factor;Demmel rus 281

			stepX = stepXIn;
			stepY = stepYIn;
			subsupra = stepX * stepX / (stepY * stepY);//[SNR] p.106, (4)
			ai = subsupra;//subdiagonal,[SNR] p.106, (4)
			bi = subsupra;//supradiagonal,[SNR] p.106, (4)

			fKsi = fKsiIn;

			this.lstBitmap = lstBitmap;

			reportProgress = reportProgressIn;
			curProgress = 0;
			oldProgress = 0;

			cBase = (T.One + subsupra) * _2;//[SNR] p.106, (4)

			this.cCores = cCores;
			optionsParallel = new ParallelOptions() { MaxDegreeOfParallelism = cCores, CancellationToken = cts.Token };

			if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
			{
				plans = new Dictionary<int, CLFFTPlan>();
				fft = new FFTOCL<T>(commands, plans, Ny, true);
			}
			else fft = new FFTLomontOCL<T>(commands, Ny);

			fftInOutSize = (Ny / 2 + 1) * FFTConstant.sizeOfComplex;
			try
			{
				fn = new T[dim1Fn * dim2Fn];
				if (lstBitmap != null) unShow = new T[Nx + 1, Ny + 1];
				pq0 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, L2 * fftInOutSize);
				pq1 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, L2 * fftInOutSize);
				pq2 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, L2 * fftInOutSize);
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
		}

		public T doIteration(int iter)
		{
			initElapsedList();
			float elapsed = getExecutedSeconds(stopWatchEL, () => initFj());
			listElapsedAdd("initFj", elapsed);

			try
			{
				forwardPath();
				if (iterationsCanceled) return T.Zero;

				solveReducedSystem();
				if (iterationsCanceled) return T.Zero;

				reversePath();
				if (iterationsCanceled) return T.Zero;

				un = new T[(Nx + 1) * (Ny + 1)];
				commands.ReadFromBuffer(unOCL, ref un, true, null);

				elapsed = getExecutedSeconds(stopWatchEL, () => restoreBounds(un));
				listElapsedAdd("restoreBounds", elapsed);
			}
			catch (Exception ex)
			{
				UtilsThread.runOnUIThreadAsync(() => MessageBox.Show(ex.Message, ex.TargetSite.Name));
			}

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		void initFj()
		{
			T stepX2 = stepX * stepX;
			if (fKsi != null) GridIterator.iterate(0, Nx - 1, 1, Ny, (i, j) => { fn[i * dim2Fn + j] += (stepX2 * fKsi(stepX * T.CreateTruncating(i + 1), stepY * T.CreateTruncating(j))); });//[SNR] p.123 (8),[SNE] p.120 (8)

			Parallel.For(0, Nx - 1, optionsParallel, (i) =>
			{
				fn[i * dim2Fn + 1] += (subsupra * bndB[i]);//[SNR] p.123 (8),[SNE] p.120 (8)
				fn[i * dim2Fn + Ny - 1] += (subsupra * bndT[i]);//[SNR] p.123 (8),[SNE] p.120 (8)
			});

			Parallel.For(1, Ny, optionsParallel, (j) =>
			{
				fn[0 * dim2Fn + j] += bndL[j - 1];//analog [SNR] p.123 (8),[SNE] p.120 (8)
				fn[(Nx - 2) * dim2Fn + j] += bndR[j - 1];//analog [SNR] p.123 (8),[SNE] p.120 (8)
			});

			fnOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, fn);
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
			KernelOCL kernelCalculatePkQk = MarchingKernelCalculatePQVectorsOCL<T>.createKernelCalculatePQVectors(commands.Device, commands.Context, L, k, Nx, M, fftInOutSize, dim2Fn, fKsi == null);
			kernelCalculatePkQk.SetMemoryArgument(0, pq0);
			kernelCalculatePkQk.SetMemoryArgument(1, pq1);
			kernelCalculatePkQk.SetMemoryArgument(2, pq2);
			kernelCalculatePkQk.SetMemoryArgument(3, fnOCL);
			kernelCalculatePkQk.SetValueArgument(4, cBase);
			kernelCalculatePkQk.SetValueArgument(5, ai);
			kernelCalculatePkQk.SetValueArgument(6, bi);

			float elapsed = getExecutedSeconds(stopWatchEL, () =>
			{
				commands.Execute(kernelCalculatePkQk, null, new long[] { L2 }, null, null);
				commands.Finish();
			});
			listElapsedAdd("calculatePQVectors", elapsed);
			UtilsCL.disposeKP(ref kernelCalculatePkQk);
			//UtilsCL.printOCLBuffer(fnOCL, commands, dim1Fn, dim2Fn, string.Format("{0} fn", commands.Device.Name));
			//UtilsCL.printOCLBuffer(pq1, commands, L2, fftInOutSize, string.Format("{0} pq1", commands.Device.Name));
			//UtilsCL.printOCLBuffer(pq2, commands, L2, fftInOutSize, string.Format("{0} pq2", commands.Device.Name));
		}

		void calculateĜ()
		{
			swapPQArrays();
			float elapsed;

			KernelOCL kernelPrepareFFT = MarchingKernelPrepareFFTOCL<T>.createKernelPrepareFFT(commands.Device, commands.Context, L, Ny, fftInOutSize);
			kernelPrepareFFT.SetMemoryArgument(0, pq0);
			kernelPrepareFFT.SetMemoryArgument(1, pq1);
			kernelPrepareFFT.SetMemoryArgument(2, pq2);
			//UtilsCL.printOCLBuffer(pq1, commands, L2, fftInOutSize, string.Format("{0} pq1", commands.Device.Name));
			//UtilsCL.printOCLBuffer(pq2, commands, L2, fftInOutSize, string.Format("{0} pq2", commands.Device.Name));

			elapsed = getExecutedSeconds(stopWatchEL, () =>
			{
				commands.Execute(kernelPrepareFFT, null, new long[] { L2, M }, null, null);
				commands.Finish();
			});
			listElapsedAdd("calculateĜ", elapsed);
			//UtilsCL.printOCLBuffer(pq0, commands, L2, fftInOutSize, string.Format("{0} pq0", commands.Device.Name));

			UtilsCL.disposeKP(ref kernelPrepareFFT);
			pq1?.Dispose();
			pq1 = null;
			pq2?.Dispose();
			pq2 = null;

			//UtilsCL.printOCLBuffer(pq0, commands, L2, fftInOutSize, string.Format("{0} pq0", "before FFT"));
			elapsed = getExecutedSeconds(stopWatchEL, () => fft.calculate(pq0/*bufFFT*/, L2, normFactor));
			listElapsedAdd("fft 1", elapsed);
			//UtilsCL.printOCLBuffer(pq0, commands, L2, fftInOutSize, string.Format("{0} pq0", "after FFT"));
		}

		void swapPQArrays()
		{
			int rem = (k - 1) % 3;
			if (rem == 1)
			{//tmp <- 0 <- 1 <- 2 <- tmp
				BufferOCL<T> tmp = pq0;
				pq0 = pq1;
				pq1 = pq2;
				pq2 = tmp;
			}
			else if (rem == 2)
			{//tmp <- 2 <- 1 <- 0 <- tmp
				BufferOCL<T> tmp = pq2;
				pq2 = pq1;
				pq1 = pq0;
				pq0 = tmp;
			}
		}

		void solveReducedSystem()
		{
			BufferOCL<T> φ = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, M * L2);
			BufferOCL<T> α = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, M * L);

			KernelOCL kernelReducedSystem = MarchingKernelReducedSystemOCL<T>.createKernelReducedSystem(commands.Device, commands.Context, L, k, M, fftInOutSize);
			kernelReducedSystem.SetMemoryArgument(0, φ);
			kernelReducedSystem.SetMemoryArgument(1, α);
			kernelReducedSystem.SetMemoryArgument(2, pq0);
			kernelReducedSystem.SetValueArgument(3, T.Pi / T.CreateTruncating(2 * Ny));
			kernelReducedSystem.SetValueArgument(4, subsupra * _2);
			//UtilsCL.printOCLBuffer(pq0, commands, L2, fftInOutSize, "pq0 before");

			float elapsed = getExecutedSeconds(stopWatchEL, () =>
			{
				commands.Execute(kernelReducedSystem, null, new long[] { M }, null, null);
				commands.Finish();
			});
			listElapsedAdd("ReducedSystem", elapsed);
			showProgress(20);

			//UtilsCL.printOCLBuffer(φ, commands, M, L2, "φ");
			//UtilsCL.printOCLBuffer(α, commands, M, L, "α");
			//UtilsCL.printOCLBuffer(pq0, commands, L2, fftInOutSize, "pq0 after");
			//UtilsCU.printCUDABuffer(φ, M, L2, "φ");
			//UtilsCU.printCUDABuffer(ŵ, L2, fftInOutSize, "ŵ");
			UtilsCL.disposeKP(ref kernelReducedSystem);
			φ?.Dispose();
			φ = null;
			α?.Dispose();
			α = null;
		}

		void reversePath()
		{
			float elapsed = 0;

			elapsed = getExecutedSeconds(stopWatchEL, () => fft.calculate(pq0/*bufFFT*/, L2, normFactor));
			listElapsedAdd("fft 2", elapsed);

			unOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, (Nx + 1) * (Ny + 1));
#if DEBUG
			UtilsCL.initBuffer<T>(unOCL, commands, T.Zero);
#endif
			//UtilsCL.printOCLBuffer(unOCL, commands, Nx + 1, Ny + 1, string.Format("{0} un after create", commands.Device.Name));
			//UtilsCL.printOCLBuffer(pq0, commands, L2, fftInOutSize, string.Format("{0} pq0", commands.Device.Name));

			KernelOCL kernelPlaceWToUn = MarchingKernelPlaceWToUnOCL<T>.createKernelPlaceWToUn(commands.Device, commands.Context, L, k, M, Ny + 1, fftInOutSize);
			kernelPlaceWToUn.SetMemoryArgument(0, pq0);
			kernelPlaceWToUn.SetMemoryArgument(1, unOCL);

			elapsed = getExecutedSeconds(stopWatchEL, () =>
			{
				commands.Execute(kernelPlaceWToUn, null, new long[] { L2 }, null, null);
				commands.Finish();
			});
			listElapsedAdd("PlaceWToUn", elapsed);

			if (unShow != null)
			{
				SysIntX2 offsSrc = new SysIntX2(1, 0), offsDst = new SysIntX2(1, 0), region = new SysIntX2(Ny - 1, 1);
				for (int iW = 0; iW < 2 * L; iW ++)
				{
					int iUN = iW * k + ((iW & 1) != 1 ? 1 : k);
					offsSrc.Y = (IntPtr)iUN;
					offsDst.Y = (IntPtr)iUN;
					commands.ReadFromBuffer(unOCL, ref unShow, true, offsDst, offsSrc, region, (Ny + 1) * Marshal.SizeOf(typeof(T)), (Ny + 1) * Marshal.SizeOf(typeof(T)), null);
				}
				UtilsPict.addPicture(lstBitmap, true, null/*minMax*/, new Adapter2D<float>(Nx + 1, Ny + 1, (m, k) => float.CreateTruncating(unShow[m, k])), fCreateBitmap);
			}

			//UtilsCU.printCUDABuffer(unCU, Nx + 1, Ny + 1, "unCU");
			UtilsCL.disposeKP(ref kernelPlaceWToUn);
			pq0?.Dispose();
			pq0 = null;

			calculateY();
			showProgress(20);
		}

		void calculateY()
		{
			KernelOCL kernelCalculateY = MarchingKernelCalculateYOCL<T>.createKernelCalculateY(commands.Device, commands.Context, L, k, Nx, M, Ny + 1, dim2Fn, fKsi == null);
			kernelCalculateY.SetMemoryArgument(0, unOCL);
			kernelCalculateY.SetMemoryArgument(1, fnOCL);
			kernelCalculateY.SetValueArgument(2, cBase);
			kernelCalculateY.SetValueArgument(3, ai);
			kernelCalculateY.SetValueArgument(4, bi);

			//UtilsCL.printOCLBuffer(fnOCL, commands, dim1Fn, dim2Fn, string.Format("{0} fn before calculateY", commands.Device.Name));
			//UtilsCL.printOCLBuffer(unOCL, commands, Nx + 1, Ny + 1, string.Format("{0} un before calculateY", commands.Device.Name));
			float elapsed = getExecutedSeconds(stopWatchEL, () =>
			{
				commands.Execute(kernelCalculateY, null, new long[] { L2 }, null, null);
				commands.Finish();
			});
			listElapsedAdd("CalculateY", elapsed);

			//UtilsCU.printCUDABuffer(unCU, Nx + 1, Ny + 1, "unCU final");
			UtilsCL.disposeKP(ref kernelCalculateY);
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
				T nan = T.Zero / T.Zero;
				GridIterator.iterate(0, Nx + 1, 0, Ny + 1, (i, j) => unShow[i, j] = nan);
				fillBounds(unShow);

				void fillBounds(T[,] uu)
				{
					Parallel.For(1, Nx, optionsParallel, (i) =>
					{
						uu[i, 0] = bndB[i - 1];
						uu[i, Ny] = bndT[i - 1];
					});

					Parallel.For(1, Ny, optionsParallel, (j) =>
					{
						uu[0, j] = bndL[j - 1];
						uu[Nx, j] = bndR[j - 1];
					});

					uu[0, 0] = nan;
					uu[Nx, 0] = nan;
					uu[0, Ny] = nan;
					uu[Nx, Ny] = nan;
				}
			}
		}
		public int maxIterations() { return 1; }
		public bool shouldReportProgress() { return false; }
		public void cancelIterations() { iterationsCanceled = true; }

		public virtual void cleanup()
		{
			UtilsCL.disposeBuf(ref pq0);
			UtilsCL.disposeBuf(ref pq1);
			UtilsCL.disposeBuf(ref pq2);

			UtilsCL.disposeBuf(ref unOCL);
			UtilsCL.disposeBuf(ref fnOCL);

			if (plans != null) foreach (var plan in plans) plan.Value.Destroy();
			plans = null;
			fft?.cleanup();
			fft = null;
			un = null;

			UtilsCL.disposeQC(ref commands);
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

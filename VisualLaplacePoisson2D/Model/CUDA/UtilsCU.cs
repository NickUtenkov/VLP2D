using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;

namespace VLP2D.Common
{
	public static class UtilsCU
	{
		static public string kernelPrefix = "\n\nextern \"C\" __global__ void\n";
		public struct DeviceCU
		{
			static string[] namesToRemove = { "NVIDIA ", "GeForce ", " Laptop", " GPU" };
			public String name { get; set; }//property because of binding in taskItemTemplate
			public DeviceCU(string devName)
			{
				foreach (string firmName in namesToRemove)
				{
					devName = devName.Replace(firmName, string.Empty);
				}
				name = devName;
			}
		}

		public static DeviceCU[] getCUDADevices()
		{
			List<DeviceCU> listDev = new List<DeviceCU>();

			try
			{
				for (int i = 0; i < CudaContext.GetDeviceCount(); i++)
				{
					CudaDeviceProperties deviceProp = CudaContext.GetDeviceInfo(i);
					if (deviceProp != null)
					{
						listDev.Add(new DeviceCU(deviceProp.DeviceName));
					}
				}
			}
			catch (Exception)
			{
				listDev.Add(new DeviceCU("NVIDIA Test CUDA device"));
			}

			return listDev.ToArray();
		}

		public static CudaKernel createKernel(string srcProgram, string kernelName, CudaContext ctx)
		{
			byte[] ptx = createPTX(srcProgram, kernelName);
			return ctx.LoadKernelPTX(ptx, kernelName);
		}

		static byte[] createPTX(string srcProgram, string kernelName)
		{
			//Debug.WriteLine(srcProgram);
			string[] options = new string[]
#if DEBUG
				{ "-arch=sm_86" }//, "--fmad=false" , "--device-debug"
#else
				{ "-arch=sm_86" }//, "--fmad=false"
#endif
			;

			CudaRuntimeCompiler rtc = new CudaRuntimeCompiler(srcProgram, kernelName);

			byte[] ptx = null;
			try
			{
				rtc.Compile(options);//null
#if DEBUG
				ptx = rtc.GetPTX();
#else
			ptx = rtc.GetCubin();
#endif
			}
			catch (Exception ex)
			{
				Debug.WriteLine(string.Format("Compile exception \"{0}\"", ex.Message));
			}

			string log = rtc.GetLogAsString();
			if (log.Length > 0 && ptx == null)
			{
				Debug.WriteLine("* Log begin *");
				Debug.WriteLine(log);
				Debug.WriteLine("* Log end *");
			}

			rtc.Dispose();

			return ptx;
		}

		public static CUmodule createModule(string srcProgram, CudaContext ctx, string fileName, bool bSave = true)
		{
			byte[] bytes = createPTX(srcProgram, null);
			if (bSave && bytes != null && fileName != null)
			{
				using (BinaryWriter writer = new BinaryWriter(File.Open(createPTXPath(fileName), FileMode.Create)))
				{
					writer.Write(bytes);
				}
			}
			return ctx.LoadModulePTX(bytes);
		}

		public static string moduleName(string name, string strTypeName, long deviceId)
		{
			return name + "_" + strTypeName + "_device" + deviceId;
		}

		public static CUmodule? loadModule(string fileName, CudaContext ctx)
		{
			if (fileName == null) return null;
			string pathPTX = createPTXPath(fileName);
			FileInfo infoPTX = new FileInfo(pathPTX);

			byte[] bytes = null;
			if (infoPTX.Exists)
			{
				int size = (int)infoPTX.Length;
				using (BinaryReader reader = new BinaryReader(File.Open(pathPTX, FileMode.Open)))
				{
					bytes = reader.ReadBytes(size);
				}
			}
			if (bytes == null) return null;
			return ctx.LoadModulePTX(bytes);
		}

		static string createPTXPath(string fileName)
		{
			return Path.Combine(binPath(), fileName + ptxExtension());
		}

		static string binPath()
		{
			string dstDirectory = Path.Combine(Utils.getDataPath(), "CUDA");
			Directory.CreateDirectory(dstDirectory);

			return dstDirectory;
		}
#if DEBUG
		static string ptxExtension() => ".ptx";
#else
		static string ptxExtension() => ".cubin";
#endif

		public static void VerticalStripCopyToDevice<T>(T[] unSrc, int dim1, int dim2, CudaDeviceVariable<T> unDst, int offsDim1, int stripWidth) where T : struct
		{//https://stackoverflow.com/questions/16731013/1d-fft-transform-of-2d-array-in-cuda
			SizeT sizeInBytesToCopy = stripWidth * Marshal.SizeOf(typeof(T));
			SizeT nextLineOffset = dim2 * Marshal.SizeOf(typeof(T));
			SizeT offsetSrc = offsDim1 * Marshal.SizeOf(typeof(T)), offsetDst = 0;
			for (int j = 0; j < dim1; j++)
			{
				unDst.CopyToDevice(unSrc, offsetSrc, offsetDst, sizeInBytesToCopy);
				offsetSrc += nextLineOffset;
				offsetDst += sizeInBytesToCopy;
			}
		}

		public static void VerticalStripCopyToHost<T>(CudaDeviceVariable<T> unSrc, int dim1, int dim2, T[] unDst, int offsDim1, int stripWidth) where T : struct
		{
			SizeT sizeInBytesToCopy = stripWidth * Marshal.SizeOf(typeof(T));
			SizeT nextLineOffset = dim2 * Marshal.SizeOf(typeof(T));
			SizeT offsetSrc = 0, offsetDst = offsDim1 * Marshal.SizeOf(typeof(T));
			for (int j = 0; j < dim1; j++)
			{
				unSrc.CopyToHost(unDst, offsetSrc, offsetDst, sizeInBytesToCopy);
				offsetSrc += sizeInBytesToCopy;
				offsetDst += nextLineOffset;
			}
		}

		public static void HorizontalFFTStripCopyToDevice<T>(T[] unSrc, int fftSize, CudaDeviceVariable<T> unDst, int offsDim2, int stripHeight) where T : struct
		{
			int fftOutputSize = (fftSize / 2 + 1) * 2;//Hermitian redundancy
			int padLeft = 1;
			SizeT dstOffsetIncrementInBytes = fftOutputSize * Marshal.SizeOf(typeof(T));
			SizeT sizeInBytesToCopy = (fftSize - padLeft) * Marshal.SizeOf(typeof(T));
			SizeT offsetSrc = ((long)offsDim2) * sizeInBytesToCopy, offsetDst = padLeft * Marshal.SizeOf(typeof(T));
			for (int i = 0; i < stripHeight; i++)
			{
				unDst.CopyToDevice(unSrc, offsetSrc, offsetDst, sizeInBytesToCopy);
				offsetSrc += sizeInBytesToCopy;
				offsetDst += dstOffsetIncrementInBytes;
			}
		}

		public static void HorizontalFFTStripCopyToHost<T>(CudaDeviceVariable<T> unSrc, int fftSize, T[] unDst, int offsDim2, int stripHeight) where T : struct
		{
			int fftInputSize = (fftSize / 2 + 1) * 2;//Hermitian redundancy
			int padLeft = 1;
			SizeT srcOffsetIncrementInBytes = fftInputSize * Marshal.SizeOf(typeof(T));
			SizeT sizeInBytesToCopy = (fftSize - padLeft) * Marshal.SizeOf(typeof(T));
			SizeT offsetSrc = padLeft * Marshal.SizeOf(typeof(T));
			SizeT offsetDst = ((long)offsDim2) * sizeInBytesToCopy;
			for (int i = 0; i < stripHeight; i++)
			{
				unSrc.CopyToHost(unDst, offsetSrc, offsetDst, sizeInBytesToCopy);
				offsetSrc += srcOffsetIncrementInBytes;
				offsetDst += sizeInBytesToCopy;
			}
		}

		public static float getExecutedMiSeconds(CudaEvent evtStart, CudaEvent evtStop, Action action)
		{
			evtStart.Record();
			action();
			evtStop.Record();
			evtStop.Synchronize();

			return CudaEvent.ElapsedTime(evtStart, evtStop);
		}

		public static CudaKernel createEpsExceededKernel<T>(int dimX, int dimY, CudaContext ctx)
		{
			string functionName = "EpsExceeded";
			string args = string.Format("({0} *un0, {0} *un1, int *flag, int shiftDim1, {0} eps)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceEpsExceeded =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1 + shiftDim1;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int idx = i * {1} + j;

	if (i <= {0} - 2 && j <= {1} - 2)
	{{
		if (fabs(un0[idx] - un1[idx]) > eps) flag[0] = 1;
	}}
}}";
			string strProgram0 = string.Format(programSourceEpsExceeded, dimX, dimY);
			string strProgram = strProgramHeader + strProgram0;
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

			return createKernel(strProgram, functionName, ctx);
		}

		public static string createEpsExceededProgram<T>(int dimX, int dimY, string functionName)
		{
			string args = string.Format("({0} *un0, {0} *un1, int *flag, int shiftDim1, {0} eps)", Utils.getTypeName<T>());
			string strProgramHeader = UtilsCU.kernelPrefix + functionName + args;
			string programSourceEpsExceeded =
@"
{{
	int i = blockDim.x * blockIdx.x + threadIdx.x + 1 + shiftDim1;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
	int idx = i * {1} + j;

	if (i <= {0} - 2 && j <= {1} - 2)
	{{
		if (fabs(un0[idx] - un1[idx]) > eps) flag[0] = 1;
	}}
}}";
			string strProgram0 = string.Format(programSourceEpsExceeded, dimX, dimY);
			return strProgramHeader + strProgram0;
		}

		public static void initBuffer(CudaDeviceVariable<float> bufCU, float val)
		{
			float[] bufFloat = new float[bufCU.Size];
			for (int i = 0; i < bufFloat.Length; i++) bufFloat[i] = val;
			bufCU.CopyToDevice(bufFloat);
		}

		public static void printCUDABuffer<T>(CudaDeviceVariable<T> ar, int dim1, int dim2, string header) where T : struct
		{
#if DEBUG
			Debug.WriteLine(header);
			for (int i = 0; i < dim1; i++)
			{
				int j = 0;
				string s1 = string.Format(CultureInfo.InvariantCulture, "{0,9:0.000}", ar[i * dim2 + j]);
				for (j = 1; j < dim2; j++) s1 += string.Format(CultureInfo.InvariantCulture, " {0,9:0.000}", ar[i * dim2 + j]);
				Debug.WriteLine(s1);
			}
#endif
		}

		public static void set1DKernelDims(CudaKernel kernel, int dim)
		{
			int threadsPerBlock = 256;// kernel.GetOccupancyMaxPotentialBlockSize().blockSize;//kernel.MaxThreadsPerBlock;
			int blocksPerDim = (dim + threadsPerBlock - 1) / threadsPerBlock;
			kernel.BlockDimensions = new dim3(threadsPerBlock, 1, 1);
			kernel.GridDimensions = new dim3(blocksPerDim, 1, 1);
			Debug.WriteLine(string.Format("{0} threadsPerBlock {1} blocksPerDim {2}", kernel.KernelName, threadsPerBlock, blocksPerDim));
		}

		public static void set1DKernelDimsPow2(CudaKernel kernel, int dim, int threadsPerBlock)
		{
			int blocksPerDim = (dim + threadsPerBlock - 1) / threadsPerBlock;
			kernel.BlockDimensions = new dim3(threadsPerBlock, 1, 1);
			kernel.GridDimensions = new dim3(blocksPerDim, 1, 1);
			Debug.WriteLine(string.Format("{0} threadsPerBlock {1} blocksPerDim {2}", kernel.KernelName, threadsPerBlock, blocksPerDim));
		}

		public static void set2DKernelDims(CudaKernel kernel, int dimX, int dimY)
		{
			int threadsPerBlockDim1 = (int)Math.Round(Math.Sqrt(kernel.MaxThreadsPerBlock));
			int threadsPerBlockDim2 = threadsPerBlockDim1;
			int blocksPerDim1 = (dimX + threadsPerBlockDim1 - 1) / threadsPerBlockDim1;
			int blocksPerDim2 = (dimY + threadsPerBlockDim2 - 1) / threadsPerBlockDim2;
			kernel.BlockDimensions = new dim3(threadsPerBlockDim1, threadsPerBlockDim2, 1);
			kernel.GridDimensions = new dim3(blocksPerDim1, blocksPerDim2, 1);
		}

		public static void set2DKernelDims(CudaKernel kernel, int dimX, int dimY, int threadsPerBlockDim1, int threadsPerBlockDim2)
		{
			int blocksPerDim1 = (dimX + threadsPerBlockDim1 - 1) / threadsPerBlockDim1;
			int blocksPerDim2 = (dimY + threadsPerBlockDim2 - 1) / threadsPerBlockDim2;
			kernel.BlockDimensions = new dim3(threadsPerBlockDim1, threadsPerBlockDim2, 1);
			kernel.GridDimensions = new dim3(blocksPerDim1, blocksPerDim2, 1);
		}

		public static SizeT getMemUsed(CudaContext ctx)
		{
			return ctx.GetTotalDeviceMemorySize() - ctx.GetFreeDeviceMemorySize();
		}

		public static int getMaxThreads(CudaContext ctx)
		{
			CudaDeviceProperties props = ctx.GetDeviceInfo();
			Version ver = props.ComputeCapability;
			return props.MultiProcessorCount * ConvertSMVer2Cores(ver.Major, ver.Minor);
		}

		public static void disposeContext(ref CudaContext ctx)
		{
			ctx?.Dispose();
			ctx = null;
		}

		public static void disposeBuf<T>(ref CudaDeviceVariable<T> buf) where T : struct
		{
			buf?.Dispose();
			buf = null;
		}

		static int ConvertSMVer2Cores(int major, int minor)
		{
			// Returns the number of CUDA cores per multiprocessor for a given
			// Compute Capability version. There is no way to retrieve that via
			// the API, so it needs to be hard-coded.
			// See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
			switch ((major << 4) + minor)
			{
				case 0x10: return 8;    // Tesla
				case 0x11: return 8;
				case 0x12: return 8;
				case 0x13: return 8;
				case 0x20: return 32;   // Fermi
				case 0x21: return 48;
				case 0x30: return 192;  // Kepler
				case 0x32: return 192;
				case 0x35: return 192;
				case 0x37: return 192;
				case 0x50: return 128;  // Maxwell
				case 0x52: return 128;
				case 0x53: return 128;
				case 0x60: return 64;   // Pascal
				case 0x61: return 128;
				case 0x62: return 128;
				case 0x70: return 64;   // Volta
				case 0x72: return 64;   // Xavier
				case 0x75: return 64;   // Turing
				case 0x80: return 64;   // Ampere
				case 0x86: return 128;
				case 0x87: return 128;
				case 0x89: return 128;  // Ada
				case 0x90: return 129;  // Hopper
				default: return 0;
			}
		}
	}
}

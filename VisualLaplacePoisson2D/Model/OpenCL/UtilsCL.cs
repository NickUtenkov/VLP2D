using Cloo;
using DD128Numeric;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;

namespace VLP2D.Common
{
	public static class UtilsCL
	{
		static public string kernelPrefix = "\n\nkernel void\n";
		public struct PlatformDevice
		{
			static string[] namesToRemove = { "Intel(R) Core(TM) ", "Intel(R) ", "Iris(R) ", "NVIDIA ", "GeForce ", " Laptop", " GPU" };
			public PlatformOCL platform;
			public DeviceOCL device;
			public String name { get; set; }//property because of binding in taskItemTemplate
			public PlatformDevice(PlatformOCL platform, DeviceOCL device)
			{
				this.platform = platform;
				this.device = device;
				string devName = device.Name;
				foreach (string firmName in namesToRemove)
				{
					devName = devName.Replace(firmName, string.Empty);
				}
				name = devName;
			}
		}

		public static PlatformDevice[] getGPUDevices()
		{
			List<PlatformDevice> listDev = new List<PlatformDevice>();
			foreach (PlatformOCL platform in PlatformOCL.Platforms)
			{
				foreach (DeviceOCL device in platform.Devices)
				{
					listDev.Add(new PlatformDevice(platform, device));
				}
			}

			return listDev.ToArray();
		}

		public static CommandQueueOCL createCommandQueue(PlatformOCL platform, DeviceOCL device, CommandQueueFlagsOCL flags)
		{
			ContextPropertyListOCL properties = new ContextPropertyListOCL(platform);
			ContextOCL context = new ContextOCL([device], properties, null, IntPtr.Zero);
			return new CommandQueueOCL(context, device, flags);
		}

		public static ProgramOCL createProgram(string strProgram, string strOptions, ContextOCL context, DeviceOCL device)
		{
			ProgramOCL program = new ProgramOCL(context, strProgram);
			string strLog = program.Build1(device, strOptions, null, IntPtr.Zero);//see options in https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#compiler-options
			if (strLog != null)
			{
				Debug.WriteLine("* Log begin *");
				Trace.WriteLine(strLog);
				Debug.WriteLine("* Log end *");
				return null;
			}
			return program;
		}

		public static ProgramOCL buildProgram(ProgramOCL program, string strOptions, ContextOCL context, DeviceOCL device)
		{
			string strLog = program.Build1(device, strOptions, null, IntPtr.Zero);//see options in https://www.khronos.org/registry/OpenCL/specs/2.2/html/OpenCL_API.html#compiler-options
			if (strLog != null)
			{
				Trace.WriteLine(strLog);
				return null;
			}
			return program;
		}

		public static ProgramOCL loadAndBuildProgram(string fileName, string options, ContextOCL context, DeviceOCL device)
		{
			string pathBin = createBinPath(fileName);
			FileInfo infoBin = new FileInfo(pathBin);

			byte[] bytes = null;
			if (infoBin.Exists)
			{
				int size = (int)infoBin.Length;
				using (BinaryReader reader = new BinaryReader(File.Open(pathBin, FileMode.Open)))
				{
					bytes = reader.ReadBytes(size);
				}
			}
			if (bytes == null) return null;

			ProgramOCL program = new ProgramOCL(context, [bytes], [device]);
			if (program != null) program = UtilsCL.buildProgram(program, options, context, device);

			return program;
		}

		public static void saveProgram(string fileName, byte[] bytes)
		{
			string pathBin = createBinPath(fileName);
			if (bytes != null)
			{
				using (BinaryWriter writer = new BinaryWriter(File.Open(pathBin, FileMode.Create)))
				{
					writer.Write(bytes);
				}
			}
		}

		public static string createBinPath(string fileName)
		{
			return Path.Combine(binPath(), fileName + binExtension());
		}

		static string binPath()
		{
			string dstDirectory = Path.Combine(Utils.getDataPath(), "OCL");
			Directory.CreateDirectory(dstDirectory);

			return dstDirectory;
		}
		static string binExtension() => ".bin";

		public static string programName(string name, string strTypeName, long vendorId)
		{
			return name + "_" + strTypeName + "_vendor" + vendorId;
		}

		public static void setKernelArguments<T>(KernelOCL kernel, object[] args) where T : struct
		{
			for (int k = 0; k < args.Length; k++)
			{
				object obj = args[k];
				if (obj == null) continue;
				if (obj is int) kernel.SetValueArgument(k, (int)obj);
				else if (obj is float) kernel.SetValueArgument(k, (float)obj);
				else if (obj is double) kernel.SetValueArgument(k, (double)obj);
				else if (obj is T) kernel.SetValueArgument(k, (T)obj);
				else if (obj is BufferOCL<int>) kernel.SetMemoryArgument(k, (BufferOCL<int>)obj);
				else if (obj is BufferOCL<T>) kernel.SetMemoryArgument(k, (BufferOCL<T>)obj);
				else throw new ArgumentException(string.Format("setKernelArguments not supported type {0}", obj.GetType()));
			}
		}

		public static void disposeBuf<T>(ref BufferOCL<T> buf) where T : struct
		{
			buf?.Dispose();
			buf = null;
		}

		public static void disposeKP(ref KernelOCL kernel, bool withProgram = true)//dispose Kernel & Program
		{
			if (withProgram) kernel?.Program.Dispose();
			kernel?.Dispose();
			kernel = null;
		}

		public static void disposeQC(ref CommandQueueOCL queue)//dispose Queue & Context
		{
			queue?.Context.Dispose();
			queue?.Dispose();
			queue = null;
		}

		public static long maxMemoryAllocationSize(DeviceOCL device)
		{
			long rc = device.MaxMemoryAllocationSize;
			if (device.HostUnifiedMemory) rc /= 10;//workaround for Iris Xe
			return rc;
		}

		public static void kernelInfo(DeviceOCL device, KernelOCL kernel)
		{
			long localMemSize = kernel.GetLocalMemorySize(device);
			long[] grpCompileSize = kernel.GetCompileWorkGroupSize(device);
			long prefGrpSize = kernel.GetPreferredWorkGroupSizeMultiple(device);
			long privMemSize = kernel.GetPrivateMemorySize(device);
			long grpSize = kernel.GetWorkGroupSize(device);
		}

		public static void kernelProgramInfo(KernelOCL kernel)
		{
			ProgramOCL program = kernel.Program;
			ReadOnlyCollection<byte[]> binaries = program.Binaries;
			Debug.WriteLine(kernel.FunctionName);
			foreach (byte[] bin in binaries) Debug.WriteLine(string.Format("Length {0}", bin.Length));
		}

		public static void initBuffer<T>(BufferOCL<T> bufOCL, CommandQueueOCL commands, T val) where T : struct
		{
			T[] bufFloat = new T[bufOCL.Count];
			for (int i = 0; i < bufFloat.Length; i++) bufFloat[i] = val;
			commands.WriteToBuffer(bufFloat, bufOCL, true, null);
		}

		public static void printOCLBuffer<T>(BufferOCL<T> bufOCL, CommandQueueOCL commands, int rows, int cols, string header) where T : struct
		{
#if DEBUG
			T[] bufFloat = new T[bufOCL.Count];
			commands.ReadFromBuffer(bufOCL, ref bufFloat, true, null);
			UtilsPrint.printArray1DAs2D(bufFloat, rows, cols, header);
#endif
		}

		public static string eventsTimings(List<EventBaseOCL> events)
		{
			if (events == null) return "";
			var eventTypes = new Dictionary<CommandTypeOCL, string>
			{
				{ CommandTypeOCL.NDRangeKernel, "Kernel" },
				{ CommandTypeOCL.ReadBuffer, "ReadBuf" },
				{ CommandTypeOCL.WriteBuffer, "WriteBuf" },
				{ CommandTypeOCL.ReadBufferRectangle, "ReadBufRect" },
				{ CommandTypeOCL.WriteBufferRectangle, "WriteBufRect" },
				{ CommandTypeOCL.CopyBuffer, "CopyBuffer" },
			};
			int cKernel = 0, cRead = 0, cWrite = 0, cCopy = 0;
			float durKernel = 0, durRead = 0, durWrite = 0, durCopy = 0;

			string rc = "";
			long totalNS = 0;
			for (int i = 0; i < events.Count; i++) totalNS += events[i].FinishTime - events[i].StartTime;

			for (int i = 0; i < events.Count; i++)
			{
				float duration = events[i].FinishTime - events[i].StartTime;
				string evtType = eventTypes.ContainsKey(events[i].Type) ? eventTypes[events[i].Type] : "?";
				if (events.Count < 40)
				{
					rc += String.Format("{0} {1:0.###} sec. {2:0.#}%", evtType, duration / 1E+9, 100.0 * duration / totalNS);
					rc += "\n";
				}
				if (events[i].Type == CommandTypeOCL.NDRangeKernel)
				{
					cKernel++;
					durKernel += duration;
				}
				if (events[i].Type == CommandTypeOCL.ReadBuffer || events[i].Type == CommandTypeOCL.ReadBufferRectangle)
				{
					cRead++;
					durRead += duration;
				}
				if (events[i].Type == CommandTypeOCL.WriteBuffer || events[i].Type == CommandTypeOCL.WriteBufferRectangle)
				{
					cWrite++;
					durWrite += duration;
				}
				if (events[i].Type == CommandTypeOCL.CopyBuffer)
				{
					cCopy++;
					durCopy += duration;
				}
			}

			rc += "\n";
			rc += String.Format("Kernel {0} {1:0.###} sec. {2:0.#}%", cKernel, durKernel / 1E+9, 100.0 * durKernel / totalNS);
			rc += "\n";

			rc += String.Format("Write {0} {1:0.###} sec. {2:0.#}%", cWrite, durWrite / 1E+9, 100.0 * durWrite / totalNS);
			rc += "\n";

			rc += String.Format("Read {0} {1:0.###} sec. {2:0.#}%", cRead, durRead / 1E+9, 100.0 * durRead / totalNS);
			rc += "\n";

			if (cCopy > 0)
			{
				rc += String.Format("Copy {0} {1:0.###} sec. {2:0.#}%", cCopy, durCopy / 1E+9, 100.0 * durCopy / totalNS);
				rc += "\n";
			}

			rc += String.Format("Total {0:0.###} sec.", totalNS / 1E+9);

			return rc;
		}

		public static KernelOCL createProgramEpsExceeded<T>(ContextOCL context, DeviceOCL device, BufferOCL<T> input, BufferOCL<T> output, BufferOCL<int> flag, int dimY, T eps) where T : struct
		{
			string functionName = "EpsExceeded";
			string args = string.Format("(global {0} *un0, global {0} *un1, global int *flag, {0} eps)", Utils.getTypeName<T>());
			string strProgramHeader = kernelPrefix + functionName + args;
			string programSourceEpsExceeded =
@"
{{
	int i = get_global_id(0);//indeces are 1-based, workgroup indeces are 1-based
	int j = get_global_id(1);

	int idx = i * {0} + j;

	if ({1}) flag[0] = 1;
}}";
			string strCondition = (typeof(T) == typeof(double) || typeof(T) == typeof(float)) ? "fabs(un0[idx] - un1[idx]) > eps" : "gt(fabs(sub_HH(un0[idx], un1[idx])), eps)";
			string strProgram0 = string.Format(programSourceEpsExceeded, dimY, strCondition);
			string strProgram = strProgramHeader + strProgram0;
			if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strDD128 + strProgram;
			if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionOCL.strHighPrecision_Basic + HighPrecisionOCL.strQD256 + strProgram;

			ProgramOCL programEpsExceeded = UtilsCL.createProgram(strProgram, null, context, device);
			KernelOCL kernelEpsExceeded = programEpsExceeded.CreateKernel(functionName);

			kernelEpsExceeded.SetMemoryArgument(0, input);
			kernelEpsExceeded.SetMemoryArgument(1, output);
			kernelEpsExceeded.SetMemoryArgument(2, flag);
			kernelEpsExceeded.SetValueArgument(3, eps);

			return kernelEpsExceeded;
		}

		public static void checkDeviceSupportDouble<T>(DeviceOCL device)
		{
			if (typeof(T) != typeof(float) && device.PreferredVectorWidthDouble == 0)
			{
				string str = string.Format("Device '{0}' doesn't support '{1}'", device.Name, typeof(T).Name);
				throw new Exception(str);
			}
		}
	}
}

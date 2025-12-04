using CLMathLibraries.CLFFT;
using Cloo;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace VLP2D.Model
{
	public interface IFFTOCL<T> where T : struct
	{
		void calculate(BufferOCL<T> ioData, int workSize, T coef);
		void calculateDivideByLyambdasSum(BufferOCL<T> ioData, int workSize, int offset);
		void cleanup();
	}

	internal class FFTOCL<T> : IFFTOCL<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>
	{
		protected CommandQueueOCL commands;
		Dictionary<int, CLFFTPlan> plans;
		IntPtr[] ptrsData, queues;
		ulong[] size;
		SineTransformOCL<T> sineTransform;

		public FFTOCL(CommandQueueOCL commands, Dictionary<int, CLFFTPlan> plans, int fftSize, bool useSineTransform)
		{
			this.commands = commands;
			this.plans = plans;
			queues = new IntPtr[] { commands.Handle.Value };
			size = new ulong[] { (ulong)fftSize };
			ptrsData = new IntPtr[] { IntPtr.Zero };

			if (useSineTransform)
			{
				sineTransform = new SineTransformOCL<T>(commands);
				sineTransform.createKernelPreProcess(commands.Device, commands.Context, fftSize);
				sineTransform.createKernelPostProcess(commands.Device, commands.Context, fftSize);
			}
		}

		~FFTOCL()
		{
		}

		public void calculate(BufferOCL<T> ioData, int workSize, T coef)
		{
			CLFFTPlan plan;
			if (plans.ContainsKey(workSize)) plan = plans[workSize];
			else
			{
				//supported powers 2,3,5,7,11,13
				plan = new CLFFTPlan(commands.Context.Handle.Value, CLFFTDim.CLFFT_1D, size);//CLFFT_NOTIMPLEMENTED if prime number
				plan.Layout = new Tuple<CLFFTLayout, CLFFTLayout>(CLFFTLayout.CLFFT_REAL, CLFFTLayout.CLFFT_HERMITIAN_INTERLEAVED);
				plan.ResultLocation = CLFFTResultLocation.CLFFT_INPLACE;
				plan.Precision = typeof(T) == typeof(float) ? CLFFTPrecision.CLFFT_SINGLE : CLFFTPrecision.CLFFT_DOUBLE;
				ulong fftOutSize = size[0] / 2 + 1;
				plan.PlanDistance = new Tuple<ulong, ulong>(fftOutSize * (ulong)FFTConstant.sizeOfComplex, fftOutSize);
				plan.BatchSize = (ulong)workSize;
				plan.Bake(queues);
				plans.Add(workSize, plan);
				Debug.WriteLine(string.Format("fftSize {0} batchSize {1} TmpBufferSize {2} ", size[0], workSize, plan.TemporaryBufferSize));
			}

			sineTransform?.preProcess(ioData, workSize);

			ptrsData[0] = ioData.Handle.Value;
			BufferOCL<T> tmpBuf = null;//if not use then progonkaMeetingDoubleFloat will be very slow
			if (plan.TemporaryBufferSize > 0) tmpBuf = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, (long)plan.TemporaryBufferSize / Marshal.SizeOf(typeof(T)));
			plan.EnqueueTransform(CLFFTDirection.CLFFT_FORWARD, 1, queues, 0, null, null, ptrsData, null, plan.TemporaryBufferSize > 0 ? tmpBuf.Handle.Value : IntPtr.Zero);
			commands.Finish();
			tmpBuf?.Dispose();

			sineTransform?.postProcess(ioData, workSize, coef);
		}

		public void calculateDivideByLyambdasSum(BufferOCL<T> ioData, int workSize, int offset)
		{
		}

		public static long temporaryBufferBytesSize(long fftSize, long batch)
		{
			return fftSize * batch * 6;
		}

		public static int maxBatchSize(long bufSizeInBytes, int fftSize)
		{
			return (int)(bufSizeInBytes / (fftSize * 6));
		}

		virtual public void cleanup()
		{
			commands = null;
			plans = null;
			ptrsData = null;
			queues = null;
			size = null;
			sineTransform?.cleanup();
			sineTransform = null;
		}
	}
}

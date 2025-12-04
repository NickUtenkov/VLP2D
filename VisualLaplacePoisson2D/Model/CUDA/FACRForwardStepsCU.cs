using ManagedCuda;
using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FACRForwardStepsCU<T> where T : struct, INumber<T>, IRootFunctions<T>
	{
		CudaContext ctx;//for UnloadKernel
		CudaKernel kernel;
		object[] args;
		CudaDeviceVariable<T> multiplied, accum, coef;
		int M2, L;
		T diagElem;

		public FACRForwardStepsCU(CudaContext ctx, CudaDeviceVariable<T> un, int dim1, int dim2, int N2, int valueL, T hYX2)
		{
			this.ctx = ctx;
			M2 = N2 >> 1;
			int maxWorkSize = M2 - 1;
			L = valueL;
			try
			{
				multiplied = new CudaDeviceVariable<T>(dim1 * maxWorkSize);
				accum = new CudaDeviceVariable<T>(dim1 * maxWorkSize);
				coef = new CudaDeviceVariable<T>(1 << (L - 1));
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			kernel = FACRForwardStepsKernelCU<T>.createKernelForwardSteps(dim1, dim2, ctx);
			kernel.SetConstantVariable("hYX2", hYX2);
			kernel.SetConstantVariable("ub1", dim1 - 1);
			kernel.SetConstantVariable("dim1", dim1);
			kernel.SetConstantVariable("dim2", dim2);
			args = [un.DevicePointer, multiplied.DevicePointer, accum.DevicePointer, coef.DevicePointer, 0, 0, maxWorkSize];
			UtilsCU.set1DKernelDims(kernel, maxWorkSize);

			diagElem = (T.One + hYX2) * T.CreateTruncating(2.0);
		}

		public void calculate(Func<bool> areIterationsCanceled)
		{
			int m = M2;

			int elemSize = Marshal.SizeOf(typeof(T));
			T[] diag = new T[1 << (L - 1)];
			for (int l = 1; l <= L; l++)
			{
				Utils.generateSqrtCoefs<T>(l - 1, (i, val) => diag[i] = diagElem + val);
				int cElems = 1 << (l - 1);
				coef.CopyToDevice(diag, 0, 0, cElems * elemSize);

				args[4] = l;
				args[5] = cElems;
				args[6] = m - 1;
				UtilsCU.set1DKernelDims(kernel, m - 1);

				kernel.Run(args);

				m >>= 1;
				if (areIterationsCanceled()) return;
			}
		}

		public void cleanup()
		{
			UtilsCU.disposeBuf(ref multiplied);
			UtilsCU.disposeBuf(ref accum);
			UtilsCU.disposeBuf(ref coef);
			ctx?.UnloadKernel(kernel);
			ctx = null;
		}
	}
}

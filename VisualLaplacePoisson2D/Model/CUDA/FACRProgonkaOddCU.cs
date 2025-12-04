using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	internal class FACRProgonkaOddCU<T> where T : struct, INumber<T>, IRootFunctions<T>, ILogarithmicFunctions<T>
	{
		CudaContext ctx;
		CudaKernel kernel;
		object[] args;
		CudaDeviceVariable<T> alphaCU, unCU;
		T[] alfa;
		int[] alfaOffsets, alfaCounts;
		CudaDeviceVariable<int> alfaOffsetsCU, alfaCountsCU;
		int N2, L, alfaSize;
		T x2DivY2;
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		T _2 = T.CreateTruncating(2);

		public FACRProgonkaOddCU(CudaContext ctx, CudaDeviceVariable<T> unCU, int progonkaSize, int N2, int paramL, T stepX2, T stepY2)
		{
			if ((progonkaSize & 1) == 1) throw new System.Exception("FACRProgonkaOddCU progonkaSize should be even");
			this.ctx = ctx;
			this.N2 = N2;
			this.unCU = unCU;
			L = paramL;
			alfaSize = progonkaSize / 2;// + 1;
			x2DivY2 = stepX2 / stepY2;

			int maxDiagElems = 1 << (L - 1);
			alfaOffsets = new int[maxDiagElems];
			alfaCounts = new int[maxDiagElems];
			alfaOffsetsCU = new CudaDeviceVariable<int>(maxDiagElems);
			alfaCountsCU = new CudaDeviceVariable<int>(maxDiagElems);

			int maxAlfa = maxAlfaArrayElements();//instead of alfaSize * maxDiagElems
			alfa = new T[maxAlfa];
			alphaCU = new CudaDeviceVariable<T>(maxAlfa);

			kernel = FACRProgonkaOddKernelCU<T>.createKernelMeetingProgonkaOdd(ctx);
			kernel.SetConstantVariable("U", progonkaSize - 1);
			kernel.SetConstantVariable("midX", progonkaSize / 2);
			kernel.SetConstantVariable("dimY", N2 - 1);
			kernel.SetConstantVariable("paramL", paramL);
			kernel.SetConstantVariable("stepX2", stepX2);
			kernel.SetConstantVariable("x2DivY2", x2DivY2);

			args = [unCU.DevicePointer, alphaCU.DevicePointer, alfaOffsetsCU.DevicePointer, alfaCountsCU.DevicePointer, 0, 0, 0];
		}

		public void calculate(Func<bool> areIterationsCanceled, int dim1, float[] unShow, List<BitmapSource> lstBitmap, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap)
		{
			for (int l = L; l >= 1; l--)
			{
				calcAlphaCU(l);

				args[4] = l;
				args[5] = 1 << (l - 1);//idxDelta
				int workSize = N2 >> l;
				args[6] = workSize;
				UtilsCU.set1DKernelDims(kernel, workSize);
				kernel.Run(args);
				if (areIterationsCanceled()) return;
				if (unShow != null)
				{
					int dim2 = N2 - 1;
					for (int jLoop = 1; jLoop <= workSize; jLoop++)
					{
						int jl = (jLoop << l) - (1 << (l - 1)) - 1;//2,6,10,14,...(for L==2)
						for (int i = 0; i < dim1; i++) unShow[i * dim2 + jl] = float.CreateTruncating(unCU[i * dim2 + jl]);
					}
					UtilsPict.addPicture(lstBitmap, true, null, new Adapter2D<float>(dim1, dim2, (m, k) => unShow[m * dim2 + k]), fCreateBitmap);
				}
			}
		}

		public void cleanup()
		{
			unCU = null;
			UtilsCU.disposeBuf(ref alphaCU);
			UtilsCU.disposeBuf(ref alfaOffsetsCU);
			UtilsCU.disposeBuf(ref alfaCountsCU);
			ctx?.UnloadKernel(kernel);
			ctx = null;
		}

		void calcAlphaCU(int curL)
		{
			void calcAlpha(T diagElem, int offset, int alfaCount)
			{
				alfa[0 + offset] = T.One / diagElem;//[SNR] p.195(40)
				for (int i = 1; i < alfaCount; i++) alfa[i + offset] = T.One / (diagElem - alfa[i - 1 + offset]);//[SNR] p.195(40)
			}
			T[] diag = new T[1 << (curL - 1)];
			Utils.generateSqrtCoefs<T>(curL - 1, (i, val) => diag[i] = val + _2);
			int offs = 0;
			for (int i = 0; i < diag.Length; i++)
			{
				T diagElem = x2DivY2 * diag[i] + _2;
				int kUp = αCC.upperBound(diagElem, alfaSize - 1) + 1;
				calcAlpha(diagElem, offs, kUp);
				alfaOffsets[i] = offs;
				alfaCounts[i] = kUp;
				offs += kUp;
			}
			alphaCU.CopyToDevice(alfa, 0, 0, offs * Marshal.SizeOf(typeof(T)));
			alfaOffsetsCU.CopyToDevice(alfaOffsets, 0, 0, diag.Length * Marshal.SizeOf(typeof(int)));
			alfaCountsCU.CopyToDevice(alfaCounts, 0, 0, diag.Length * Marshal.SizeOf(typeof(int)));
		}

		int maxAlfaArrayElements()
		{
			int maxDiagElems = 1 << (L - 1);
			T[] diag = new T[maxDiagElems];
			int maxAlfa = 0;
			for (int l = L; l >= 1; l--)
			{
				int kAll = 0;
				Utils.generateSqrtCoefs<T>(l - 1, (i, val) => diag[i] = val + _2);
				for (int i = 0; i < diag.Length; i++)
				{
					T diagElem = x2DivY2 * diag[i] + _2;
					int kUp = αCC.upperBound(diagElem, alfaSize - 1) + 1;
					kAll += kUp;
				}
				maxAlfa = Math.Max(maxAlfa, kAll);
			}
			return maxAlfa;
		}
	}
}

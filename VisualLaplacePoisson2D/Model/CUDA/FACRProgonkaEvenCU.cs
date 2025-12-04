using DD128Numeric;
using ManagedCuda;
using QD256Numeric;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class FACRProgonkaEvenCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, IMinMaxValue<T>, IRootFunctions<T>, ILogarithmicFunctions<T>
	{
		CudaContext ctx;
		CudaKernel kernelProgonka, kernelAlfaCounts;
		object[] argsProgonka, argsAlfaCounts;
		CudaDeviceVariable<T> alphaCU;
		CudaDeviceVariable<int> alfaOffsetsCU;
		CudaDeviceVariable<int> alfaCountsCU;

		public FACRProgonkaEvenCU(CudaContext ctx, CudaDeviceVariable<T> unCU, int allProgonkaWorkSize, int progonkaSize, int N2, int paramL, T stepX2, T stepY2)
		{
			bool bMeeting = true;
			if (bMeeting && (progonkaSize & 1) == 1) throw new System.Exception("FACRProgonkaEvenCU progonkaSize should be even");
			this.ctx = ctx;
			(kernelProgonka, kernelAlfaCounts) = FACRProgonkaEvenKernelCU.createKernelProgonkaEven(ctx, Utils.getTypeName<T>(), bMeeting);
			if (typeof(T) == typeof(DD128)) HighPrecisionCU.setTrogonometricConstantsDD128(kernelProgonka);
			if (typeof(T) == typeof(QD256)) HighPrecisionCU.setTrogonometricConstantsQD256(kernelProgonka);

			kernelProgonka.SetConstantVariable("U", progonkaSize - 1);//upper bound
			kernelProgonka.SetConstantVariable("midX", bMeeting ? progonkaSize / 2 : 0);//used only if bMeeting
			kernelProgonka.SetConstantVariable("dimY", N2 - 1);
			kernelProgonka.SetConstantVariable("paramL", paramL);
			kernelProgonka.SetConstantVariable("countDiagElems", (1 << paramL) - 1);
			kernelProgonka.SetConstantVariable("allProgonkaWorkSize", allProgonkaWorkSize);

			T _2 = T.CreateTruncating(2);
			T x2DivY2 = stepX2 / stepY2;
			T x2DivY2Doubled = x2DivY2 * _2;
			kernelProgonka.SetConstantVariable("stepX2", stepX2);
			kernelProgonka.SetConstantVariable("x2DivY2", x2DivY2);
			kernelProgonka.SetConstantVariable("x2DivY2Doubled", x2DivY2Doubled);
			kernelProgonka.SetConstantVariable("piDivN2", T.Pi / T.CreateTruncating(N2));
			kernelProgonka.SetConstantVariable("ck2base", x2DivY2Doubled + _2);
			kernelProgonka.SetConstantVariable("_2tm1", AlfaСonvergentUpperBound<T>.get_2tm1());

			kernelProgonka.SetConstantVariable("_x2DivY2Doubled", float.CreateTruncating(x2DivY2Doubled));
			kernelProgonka.SetConstantVariable("_piDivN2", float.CreateTruncating(T.Pi / T.CreateTruncating(N2)));
			kernelProgonka.SetConstantVariable("_ck2base", float.CreateTruncating(x2DivY2Doubled + _2));
			kernelProgonka.SetConstantVariable("_Pi4", float.CreateTruncating(T.Pi / T.CreateTruncating(4)));

			alfaCountsCU = new CudaDeviceVariable<int>(allProgonkaWorkSize);
			argsAlfaCounts = [alfaCountsCU.DevicePointer];
			UtilsCU.set1DKernelDims(kernelAlfaCounts, allProgonkaWorkSize);

			argsProgonka = [unCU.DevicePointer, null, null, alfaCountsCU.DevicePointer];
			UtilsCU.set1DKernelDims(kernelProgonka, allProgonkaWorkSize);
		}

		public void calculate()
		{
			kernelAlfaCounts.Run(argsAlfaCounts);

			int[] alfaCounts = alfaCountsCU;
			int size = alfaCountsCU.Size;
			int[] offsets = new int[size];
			int offset = 0;
			for (int i = 0; i < size; i++)
			{
				offsets[i] = offset;
				offset += alfaCounts[i];
			}

			alphaCU = new CudaDeviceVariable<T>(offset);
			argsProgonka[1] = alphaCU.DevicePointer;
			alfaOffsetsCU = offsets;
			argsProgonka[2] = alfaOffsetsCU.DevicePointer;
			kernelProgonka.Run(argsProgonka);
		}

		public void cleanup()
		{
			UtilsCU.disposeBuf(ref alphaCU);
			UtilsCU.disposeBuf(ref alfaOffsetsCU);
			UtilsCU.disposeBuf(ref alfaCountsCU);
			ctx?.UnloadModule(kernelProgonka.CUModule);
			ctx = null;
		}
	}
}

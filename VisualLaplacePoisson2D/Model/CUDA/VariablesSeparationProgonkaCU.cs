
using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using System;
using System.Diagnostics;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	public class VariablesSeparationProgonkaCU<T> where T : struct, INumber<T>, ITrigonometricFunctions<T>, IRootFunctions<T>, ILogarithmicFunctions<T>
	{
		CudaContext ctx;
		CudaKernel kernelAlfa, kernelProgonka;
		object[] argsAlfa, argsProgonka;
		T[] un;
		int dim1, dim2;
		CudaDeviceVariable<T> unCU, alfa;
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		int[] alfaOffsets, alfaCounts;
		CudaDeviceVariable<int> alfaOffsetsCU, alfaCountsCU;
		int maxProgonkaVectors, allProgonkaWorkSize;

		public VariablesSeparationProgonkaCU(CudaContext ctx, CudaDeviceVariable<T> unCU, T[] un, int dim1, int dim2, T stepX2, T stepY2, int maxProgonkaVectors, int allProgonkaWorkSize)
		{
			if ((dim1 & 1) == 1) throw new System.Exception("VariablesSeparationProgonkaCU dim1 should be even");
			this.ctx = ctx;
			this.unCU = unCU;
			this.un = un;
			this.dim1 = dim1;
			this.dim2 = dim2;
			this.maxProgonkaVectors = maxProgonkaVectors;
			this.allProgonkaWorkSize = allProgonkaWorkSize;
			int N2 = dim2 + 1;

			alfaOffsets = new int[dim2];
			alfaCounts = new int[dim2];
			int alfaAllCount = calcAlphaOffsetAndSize(T.Pi / T.CreateTruncating(N2), stepX2 / stepY2, dim1 / 2 - 1);
			alfa = new CudaDeviceVariable<T>(alfaAllCount);
			alfaOffsetsCU = alfaOffsets;
			alfaCountsCU = alfaCounts;

			CUmodule? module;
			string moduleName = UtilsCU.moduleName("VariablesSeparationProgonka_", Utils.getTypeName<T>(), ctx.DeviceId);
			string functionNameAlfa = "calcAlfa";
			string functionNameProgonka = "progonkaMeeting";

			module = UtilsCU.loadModule(moduleName, ctx);
			if (module == null)
			{
				string strProgram = VariablesSeparationProgonkaProgramCU.definesAndConstants();

				string typeName = Utils.getTypeName<T>();
				strProgram += VariablesSeparationProgonkaProgramCU.alfaProgram(functionNameAlfa, typeName);
				strProgram += VariablesSeparationProgonkaProgramCU.meetingProgonkaProgram(functionNameProgonka, typeName);

				if (typeof(T) == typeof(float)) strProgram = HighPrecisionCU.strSingleDefines + strProgram;
				if (typeof(T) == typeof(double)) strProgram = HighPrecisionCU.strDoubleDefines + strProgram;
				if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + HighPrecisionCU.strDD128Trig + strProgram;
				if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + HighPrecisionCU.strQD256Trig + strProgram;

				module = UtilsCU.createModule(strProgram, ctx, moduleName);
			}
			kernelAlfa = new CudaKernel(functionNameAlfa, (CUmodule)module);
			kernelProgonka = new CudaKernel(functionNameProgonka, (CUmodule)module);

			kernelProgonka.SetConstantVariable("U", dim1 - 1);//upper bound
			kernelProgonka.SetConstantVariable("midX", dim1 / 2);

			if (typeof(T) == typeof(DD128)) HighPrecisionCU.setTrogonometricConstantsDD128(kernelProgonka);
			if (typeof(T) == typeof(QD256)) HighPrecisionCU.setTrogonometricConstantsQD256(kernelProgonka);

			argsAlfa = [alfa.DevicePointer, alfaOffsetsCU.DevicePointer, alfaCountsCU.DevicePointer, T.Pi / T.CreateTruncating(N2), stepX2 / stepY2, allProgonkaWorkSize];
			argsProgonka = [ unCU.DevicePointer, alfa.DevicePointer, alfaOffsetsCU.DevicePointer, alfaCountsCU.DevicePointer, stepX2, 0, 0 ];
		}

		~VariablesSeparationProgonkaCU()
		{
		}

		public void calculate(Action<float> showProgress, float progressPercent, Func<bool> areIterationsCanceled)
		{
			UtilsCU.set1DKernelDims(kernelAlfa, allProgonkaWorkSize);
			kernelAlfa.Run(argsAlfa);

			int[] stripWidths = Utils.calculateWorkSizes(maxProgonkaVectors, allProgonkaWorkSize);
			int offsetJ = 0, offsSrcX = 0;

			for (int i = 0; i < stripWidths.Length; i++)
			{
				UtilsCU.VerticalStripCopyToDevice(un, dim1, dim2, unCU, offsSrcX, stripWidths[i]);
				calculateStrip(stripWidths[i], offsetJ);
				UtilsCU.VerticalStripCopyToHost(unCU, dim1, dim2, un, offsSrcX, stripWidths[i]);

				offsSrcX += stripWidths[i];
				offsetJ += stripWidths[i];

				if (areIterationsCanceled()) return;
				showProgress(progressPercent / stripWidths.Length);
			}
		}

		void calculateStrip(int workSize, int offsetJ)
		{
			try
			{
				argsProgonka[5] = offsetJ;
				argsProgonka[6] = workSize;
				UtilsCU.set1DKernelDims(kernelProgonka, workSize);
				kernelProgonka.Run(argsProgonka);
			}
			catch (Exception ex)
			{
				Debug.WriteLine(ex.Message);
			}
		}

		int calcAlphaOffsetAndSize(T pi2N2, T mult, int upper)
		{
			T _2 = T.CreateTruncating(2);
			int offset = 0;
			for (int j = 0; j < alfaCounts.Length; j++)
			{
				T cosinus = T.Cos(pi2N2 * T.CreateTruncating(j + 1));
				T diagElem = (T.One + mult * (T.One - cosinus)) * _2;
				int kUp = αCC.upperBound(diagElem, upper);
				alfaCounts[j] = kUp + 1;
				alfaOffsets[j] = offset;
				offset += alfaCounts[j];
			}
			return offset;
		}

		public void cleanup()
		{
			if (kernelProgonka != null) ctx?.UnloadModule(kernelProgonka.CUModule);
			ctx = null;
			kernelAlfa = null;
			kernelProgonka = null;
			argsProgonka = null;
			un = null;
			unCU = null;
			UtilsCU.disposeBuf(ref alfa);
			UtilsCU.disposeBuf(ref alfaOffsetsCU);
			UtilsCU.disposeBuf(ref alfaCountsCU);
		}
	}
}

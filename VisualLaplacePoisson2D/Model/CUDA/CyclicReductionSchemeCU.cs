
using DD128Numeric;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using QD256Numeric;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsElapsed;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class CyclicReductionSchemeCU<T> : CyclicReductionBaseSchemeCU<T> where T : struct, INumber<T>, IMinMaxValue<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>
	{
		CudaDeviceVariable<T> alphaCoefs, alfa, vv, diagonalElements;
		CudaDeviceVariable<T> ac, sum;
		CudaDeviceVariable<int> alfaOffsetsCU, alfaCountsCU;
		CudaKernel kernelAlfa, kernelDirect, kernelDirectSequentialSum, kernelDirectCascadeSum, kernelDirectCascadeSumBlocks, kernelPreReverse, kernelReverse, kernelReverseSequentialSum, kernelReverseCascadeSum, kernelReverseCascadeSumBlocks;
		object[] argsDirect, argsDirectSequentialSum, argsDirectCascadeSum, argsDirectCascadeSumBlocks, argsPreReverse, argsReverse, argsReverseSequentialSum, argsReverseCascadeSum, argsReverseCascadeSumBlocks;
		long[] gWorkSize = { 0 };
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		int alfaSize;
		int[] alfaOffsets, alfaCounts;

		public CyclicReductionSchemeCU(int cXSegments, int cYSegments, T stepXIn, T stepYIn, Func<T, T, T> fKsi, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn, int cudaDevice) :
			base(cXSegments, cYSegments, stepXIn, stepYIn, fKsi, lstBitmap0, fCreateBitmap, reportProgressIn, cudaDevice)
		{
			if ((N2 & 1) == 0) throw new System.Exception("CyclicReductionSchemeCU N2 should be odd");
			progressSteps = (n - 1) + n;//(n - 1) - direct steps, n - reverse steps

			alfaSize = N2 - 1;
			try
			{
				ac = new CudaDeviceVariable<T>(N1 / 2 * alfaSize);
				vv = new CudaDeviceVariable<T>(N1 / 2 * alfaSize);
				sum = new CudaDeviceVariable<T>(N1 / 2 * alfaSize);

				alphaCoefs = new CudaDeviceVariable<T>(N1 - 1);
				diagonalElements = new CudaDeviceVariable<T>(N1 - 1);

				alfaOffsets = new int[N1 - 1];
				alfaCounts = new int[N1 - 1];
				int alfaAllCount = fillAlphaOffsetAndSize();//old value = N1 / 2 * (N2 - 1)(as for ac, vv, sum)
				alfa = new CudaDeviceVariable<T>(alfaAllCount);
				alfaOffsetsCU = alfaOffsets;
				alfaCountsCU = alfaCounts;

				CUmodule? module;
				string moduleName = UtilsCU.moduleName("CyclicReduction_", Utils.getTypeName<T>(), ctx.DeviceId);
				string functionNameAlfa = "calcStoredAlfa";
				string functionNameDirect = "Direct";
				string functionNameDirectSequential = "DirectSequentialSum";
				string functionNameDirectCascade = "DirectCascadeSum";
				string functionNameDirectCascadeSumBlocks = "DirectCascadeSumBlocks";
				string functionNamePreReverse = "PreReverse";
				string functionNameReverse = "Reverse";
				string functionNameReverseSequential = "ReverseSequentialSum";
				string functionNameReverseCascade = "ReverseCascadeSum";
				string functionNameReverseCascadeSumBlocks = "ReverseCascadeSumBlocks";

				module = UtilsCU.loadModule(moduleName, ctx);
				if (module == null)
				{
					string typeName = Utils.getTypeName<T>();
					string strProgram = string.Format(CyclicReductionProgramCU.constants, typeName);

					strProgram += CyclicReductionProgramCU.createAlfaProgram(functionNameAlfa, typeName);
					strProgram += CyclicReductionProgramCU.createDirectProgram(functionNameDirect, typeName);
					strProgram += CyclicReductionProgramCU.createDirectSequentialSumProgram(functionNameDirectSequential, typeName);
					strProgram += CyclicReductionProgramCU.createDirectCascadeSumProgram(functionNameDirectCascade, typeName);
					strProgram += CyclicReductionProgramCU.createDirectCascadeSumBlocksProgram(functionNameDirectCascadeSumBlocks, typeName);
					strProgram += CyclicReductionProgramCU.createPreReverseProgram(functionNamePreReverse, typeName);
					strProgram += CyclicReductionProgramCU.createReverseProgram(functionNameReverse, typeName);
					strProgram += CyclicReductionProgramCU.createReverseSequentialSumProgram(functionNameReverseSequential, typeName);
					strProgram += CyclicReductionProgramCU.createReverseCascadeSumProgram(functionNameReverseCascade, typeName);
					strProgram += CyclicReductionProgramCU.createReverseCascadeSumBlocksProgram(functionNameReverseCascadeSumBlocks, typeName);

					if (typeof(T) == typeof(float)) strProgram = HighPrecisionCU.strSingleDefines + strProgram;
					if (typeof(T) == typeof(double)) strProgram = HighPrecisionCU.strDoubleDefines + strProgram;
					if (typeof(T) == typeof(DD128)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strDD128 + strProgram;
					if (typeof(T) == typeof(QD256)) strProgram = HighPrecisionCU.strHighPrecision_Basic + HighPrecisionCU.strQD256 + strProgram;

					module = UtilsCU.createModule(strProgram, ctx, moduleName);
				}

				kernelAlfa = new CudaKernel(functionNameAlfa, (CUmodule)module);
				kernelDirect = new CudaKernel(functionNameDirect, (CUmodule)module);
				kernelDirectSequentialSum = new CudaKernel(functionNameDirectSequential, (CUmodule)module);
				kernelDirectCascadeSum = new CudaKernel(functionNameDirectCascade, (CUmodule)module);
				kernelDirectCascadeSumBlocks = new CudaKernel(functionNameDirectCascadeSumBlocks, (CUmodule)module);
				kernelPreReverse = new CudaKernel(functionNamePreReverse, (CUmodule)module);
				kernelReverse = new CudaKernel(functionNameReverse, (CUmodule)module);
				kernelReverseSequentialSum = new CudaKernel(functionNameReverseSequential, (CUmodule)module);
				kernelReverseCascadeSum = new CudaKernel(functionNameReverseCascade, (CUmodule)module);
				kernelReverseCascadeSumBlocks = new CudaKernel(functionNameReverseCascadeSumBlocks, (CUmodule)module);

				kernelAlfa.SetConstantVariable("alfaIn", alfa.DevicePointer);
				kernelAlfa.SetConstantVariable("diagElements", diagonalElements.DevicePointer);
				kernelAlfa.SetConstantVariable("alfaOffsets", alfaOffsetsCU.DevicePointer);
				kernelAlfa.SetConstantVariable("alfaCounts", alfaCountsCU.DevicePointer);
				kernelAlfa.SetConstantVariable("alphaCoefs", alphaCoefs.DevicePointer);
				kernelAlfa.SetConstantVariable("bCoef", bCoef);
				kernelAlfa.SetConstantVariable("un", unCU.DevicePointer);
				kernelAlfa.SetConstantVariable("resIn", vv.DevicePointer);
				kernelAlfa.SetConstantVariable("sumIn", sum.DevicePointer);
				kernelAlfa.SetConstantVariable("acIn", ac.DevicePointer);

				kernelAlfa.SetConstantVariable("alfaSize", alfaSize);//N2 - 1
				kernelAlfa.SetConstantVariable("dimY", N2 + 1);
				kernelAlfa.SetConstantVariable("dimAlfa", N1 - 1);
				kernelReverse.SetConstantVariable("dimReverse", N1 / 2);

				argsDirect = new object[3];
				argsDirectSequentialSum = new object[3];
				argsDirectCascadeSum = new object[4];
				argsDirectCascadeSumBlocks = new object[5];
				argsPreReverse = new object[3];
				argsReverse = new object[2];
				argsReverseSequentialSum = new object[3];
				argsReverseCascadeSum = new object[4];
				argsReverseCascadeSumBlocks = new object[5];
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}
		}

		override public T doIteration(int iter)
		{//Q = 9.5*N2*N1*Log(N1,2) - 8*N2*N1;N2 can be == N1
			float elapsedAlfa = 0, elapsedDirect = 0, elapsedDirectCS = 0, elapsedDirectCSB = 0, elapsedDirectSS = 0,
				elapsedPreReverse = 0, elapsedReverse = 0, elapsedReverseCS = 0, elapsedReverseCSB = 0, elapsedReverseSS = 0;
			initElapsedList();

			T[] ar = new T[N1 - 1];

			fillArray(ar, alphaCoeff);
			alphaCoefs.CopyToDevice(ar);

			fillArray(ar, diagElem);
			diagonalElements.CopyToDevice(ar);

			ar = null;

			UtilsCU.set1DKernelDims(kernelReverse, N1 / 2);

			UtilsCU.set1DKernelDims(kernelAlfa, N1 - 1);
			elapsedAlfa += kernelAlfa.Run();

			int thresholdCascadeSum = 256;
			int maxThreadsCascadeSum = kernelReverseCascadeSum.MaxThreadsPerBlock;

			for (int k = 1; k <= n - 1; k++)
			{
				int m = 1 << (k - 1);
				int t = m << 1;

				argsDirect[0] = t;
				argsDirect[1] = m;
				argsDirect[2] = (N1 / t - 1) * m;
				UtilsCU.set1DKernelDims(kernelDirect, (int)argsDirect[2]);
				elapsedDirect += kernelDirect.Run(argsDirect);

				if (m > 1)
				{
					if (m >= thresholdCascadeSum)
					{
						argsDirectCascadeSum[1] = t;
						argsDirectCascadeSum[2] = m;
						int nBlocks = m / maxThreadsCascadeSum;
						argsDirectCascadeSum[3] = nBlocks;
						int threadsPerBlock = int.Min(m, maxThreadsCascadeSum);
						UtilsCU.set1DKernelDimsPow2(kernelDirectCascadeSum, (int)argsDirectCascadeSum[2], threadsPerBlock);
						for (int idx = 0; idx < N1 / t - 1; idx++)
						{
							argsDirectCascadeSum[0] = idx;
							elapsedDirectCS += kernelDirectCascadeSum.Run(argsDirectCascadeSum);

							if (nBlocks > 1)
							{
								argsDirectCascadeSumBlocks[0] = idx;
								argsDirectCascadeSumBlocks[1] = t;
								argsDirectCascadeSumBlocks[2] = m;
								argsDirectCascadeSumBlocks[3] = nBlocks;
								argsDirectCascadeSumBlocks[4] = threadsPerBlock;
								UtilsCU.set1DKernelDimsPow2(kernelDirectCascadeSumBlocks, 1, 1);
								elapsedDirectCSB += kernelDirectCascadeSumBlocks.Run(argsDirectCascadeSumBlocks);
							}
						}
					}
					else
					{
						argsDirectSequentialSum[0] = t;
						argsDirectSequentialSum[1] = m;
						argsDirectSequentialSum[2] = N1 / t - 1;
						UtilsCU.set1DKernelDims(kernelDirectSequentialSum, (int)argsDirectSequentialSum[2]);
						elapsedDirectSS += kernelDirectSequentialSum.Run(argsDirectSequentialSum);
					}
				}

				showProgress();
				if (areIterationsCanceled()) return T.Zero;
			}

			for (int k = n; k >= 1; k--)// reverse steps
			{
				int m = 1 << (k - 1);
				int t = m << 1;

				argsPreReverse[0] = t;
				argsPreReverse[1] = m;
				argsPreReverse[2] = N1 / t;
				UtilsCU.set1DKernelDims(kernelPreReverse, (int)argsPreReverse[2]);
				elapsedPreReverse += kernelPreReverse.Run(argsPreReverse);

				argsReverse[0] = t;
				argsReverse[1] = m;
				elapsedReverse += kernelReverse.Run(argsReverse);

				if (m > 1)
				{
					if (m >= thresholdCascadeSum)
					{
						argsReverseCascadeSum[1] = t;
						argsReverseCascadeSum[2] = m;
						int nBlocks = m / maxThreadsCascadeSum;
						argsReverseCascadeSum[3] = nBlocks;
						int threadsPerBlock = int.Min(m, maxThreadsCascadeSum);
						UtilsCU.set1DKernelDimsPow2(kernelReverseCascadeSum, (int)argsReverseCascadeSum[2], threadsPerBlock);
						for (int idx = 0; idx < N1 / t; idx++)
						{
							argsReverseCascadeSum[0] = idx;
							elapsedReverseCS += kernelReverseCascadeSum.Run(argsReverseCascadeSum);

							if (nBlocks > 1)
							{
								argsReverseCascadeSumBlocks[0] = idx;
								argsReverseCascadeSumBlocks[1] = t;
								argsReverseCascadeSumBlocks[2] = m;
								argsReverseCascadeSumBlocks[3] = nBlocks;
								argsReverseCascadeSumBlocks[4] = threadsPerBlock;
								UtilsCU.set1DKernelDimsPow2(kernelReverseCascadeSumBlocks, 1, 1);
								elapsedReverseCSB += kernelReverseCascadeSumBlocks.Run(argsReverseCascadeSumBlocks);
							}
						}
					}
					else
					{
						argsReverseSequentialSum[0] = t;
						argsReverseSequentialSum[1] = m;
						argsReverseSequentialSum[2] = N1 / t;
						UtilsCU.set1DKernelDims(kernelReverseSequentialSum, (int)argsReverseSequentialSum[2]);
						elapsedReverseSS += kernelReverseSequentialSum.Run(argsReverseSequentialSum);
					}
				}

				if (lstBitmap != null && k != 1) addPicture(m, N1 - m, t);//if k == 1, then picture will be the same as final(which is added in external code)

				showProgress();
				if (areIterationsCanceled()) return T.Zero;
			}

			deviceBufferToHostBuffer();

			listElapsedAdd("Alfa", elapsedAlfa / 1000);
			listElapsedAdd("Direct", elapsedDirect / 1000);
			listElapsedAdd("DirectCascSum", elapsedDirectCS / 1000);
			listElapsedAdd("DirectCascSumBlock", elapsedDirectCSB / 1000);
			listElapsedAdd("DirectSeqSum", elapsedDirectSS / 1000);
			listElapsedAdd("PreReverse", elapsedPreReverse / 1000);
			listElapsedAdd("Reverse", elapsedReverse / 1000);
			listElapsedAdd("ReverseCascSum", elapsedReverseCS / 1000);
			listElapsedAdd("ReverseCascSumBlock", elapsedReverseCSB / 1000);
			listElapsedAdd("ReverseSecSum", elapsedReverseSS / 1000);

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		override public string getElapsedInfo() { return timesElapsed(); }

		T alphaCoeff(int k, int l)
		{
			int m = 1 << (k - 1);
			T sin = T.Sin(T.Pi * T.CreateTruncating(2 * l - 1) / T.CreateTruncating(1 << k)) / T.CreateTruncating(m);
			if (l % 2 == 0) sin = -sin;
			return sin;
		}

		override public void cleanup()
		{
			UtilsCU.disposeBuf(ref alphaCoefs);
			UtilsCU.disposeBuf(ref alfa);
			UtilsCU.disposeBuf(ref ac);
			UtilsCU.disposeBuf(ref vv);
			UtilsCU.disposeBuf(ref diagonalElements);
			UtilsCU.disposeBuf(ref sum);

			if (kernelAlfa != null) ctx.UnloadModule(kernelAlfa.CUModule);
			kernelAlfa = null;
			kernelDirect = null;
			kernelReverse = null;
			kernelDirectSequentialSum = null;
			kernelDirectCascadeSum = null;
			kernelPreReverse = null;
			kernelReverseSequentialSum = null;
			kernelReverseCascadeSum = null;

			base.cleanup();
		}

		int fillAlphaOffsetAndSize()
		{
			int idx = 0;
			int offset = 0;
			for (int k = 1; k <= n; k++)
			{
				int m = 1 << (k - 1);
				for (int l = 1; l <= m; l++) // l = 1..2^{k-1}
				{
					int kUp = αCC.upperBound(diagElem(k, l), alfaSize - 1);
					alfaCounts[idx] = kUp + 1;
					alfaOffsets[idx] = offset;
					offset += alfaCounts[idx];
					idx++;
				}
			}

			return offset;
		}
	}
}

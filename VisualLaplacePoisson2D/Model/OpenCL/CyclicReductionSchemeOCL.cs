
using Cloo;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using VLP2D.Common;
using static VLP2D.Common.Utils;
using static VLP2D.Common.UtilsPict;

namespace VLP2D.Model
{
	class CyclicReductionSchemeOCL<T> : CyclicReductionBaseSchemeOCL<T> where T : struct, INumber<T>, IMinMaxValue<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>
	{
		BufferOCL<T> alphaCoefs, alfa, vv, diagonalElements;
		BufferOCL<T> ac, sum;
		BufferOCL<int> alfaOffsetsOCL, alfaCountsOCL;
		KernelOCL kernelAlfa, kernelDirect, kernelDirectSequentialSum, kernelDirectCascadeSum, kernelDirectCascadeSumBlocks, kernelPreReverse, kernelReverse, kernelReverseSequentialSum, kernelReverseCascadeSum, kernelReverseCascadeSumBlocks;
		long[] gWorkSize = { 0 }, lWorkSize = { 0 }, gWorkSizeSumBlocks = { 1 };
		AlfaСonvergentUpperBoundEpsilon αCC = new AlfaСonvergentUpperBoundEpsilon(UtilsEps.epsilon<T>());
		int alfaVectorLength;
		int[] alfaOffsets, alfaCounts;

		public CyclicReductionSchemeOCL(int cXSegments, int cYSegments, T stepXIn, T stepYIn, Func<T, T, T> fKsi, List<BitmapSource> lstBitmap0, Func<bool, MinMaxF, Adapter2D<float>, BitmapSource> fCreateBitmap, Action<double> reportProgressIn, PlatformOCL platform, DeviceOCL device) :
			base(cXSegments, cYSegments, stepXIn, stepYIn, fKsi, lstBitmap0, fCreateBitmap, reportProgressIn, platform, device)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			if ((N2 & 1) == 0) throw new System.Exception("CyclicReductionSchemeOCL N2 should be odd");
			progressSteps = (n - 1) + n;//(n - 1) - direct steps, n - reverse steps

			alfaVectorLength = N2 - 1;
			try
			{
				ac = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, N1 / 2 * alfaVectorLength);
				vv = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, N1 / 2 * alfaVectorLength);
				sum = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, N1 / 2 * alfaVectorLength);

				alphaCoefs = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, N1 - 1);
				diagonalElements = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, N1 - 1);

				alfaOffsets = new int[N1 - 1];
				alfaCounts = new int[N1 - 1];
				int alfaAllCount = fillAlphaOffsetAndSize();//old value = N1 / 2 * (N2 - 1)(as for ac, vv, sum)
				alfa = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, alfaAllCount);
				alfaOffsetsOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, alfaOffsets);
				alfaCountsOCL = new BufferOCL<int>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, alfaCounts);
			}
			catch (Exception)
			{
				cleanup();
				throw;
			}

			string typeName = Utils.getTypeName<T>();
			CyclicReductionProgramOCL kernelCreator = new CyclicReductionProgramOCL(N2, typeName, commands.Context, commands.Device);

			kernelAlfa = kernelCreator.createAlfaProgram();
			kernelDirect = kernelCreator.createDirectProgram();
			kernelDirectSequentialSum = kernelCreator.createDirectSequentialSumProgram();
			kernelDirectCascadeSum = kernelCreator.createDirectCascadeSumProgram();
			kernelDirectCascadeSumBlocks = kernelCreator.createDirectCascadeSumBlocksProgram();

			kernelPreReverse = kernelCreator.createPreReverseProgram();
			kernelReverse = kernelCreator.createReverseProgram();
			kernelReverseSequentialSum = kernelCreator.createReverseSequentialSumProgram();
			kernelReverseCascadeSum = kernelCreator.createReverseCascadeSumProgram();
			kernelReverseCascadeSumBlocks = kernelCreator.createReverseCascadeSumBlocksProgram();

			object[] argsAlfa = [alfa, alfaOffsetsOCL, alfaCountsOCL, diagonalElements];
			object[] argsDirect = [0, 0, alfa, alfaOffsetsOCL, alfaCountsOCL, alphaCoefs, bCoef, unOCL, vv];
			object[] argsDirectSeqSum = [0, 0, unOCL, vv];
			object[] argsDirectCascadeSum = [0, 0, 0, unOCL, vv, sum];
			object[] argsDirectCascadeSumBlocks = [0, 0, 0, unOCL, sum];
			object[] argsPreReverse = [0, 0, unOCL, ac];
			object[] argsReverse = [0, 0, alfa, alfaOffsetsOCL, alfaCountsOCL, alphaCoefs, bCoef, 0, unOCL, vv, ac];
			object[] argsReverseSeqSum = [0, 0, unOCL, vv];
			object[] argsReverseCascadeSum = [0, 0, 0, unOCL, vv, sum];
			object[] argsReverseCascadeSumBlocks = [0, 0, 0, unOCL, sum];

			UtilsCL.setKernelArguments<T>(kernelAlfa, argsAlfa);
			UtilsCL.setKernelArguments<T>(kernelDirect, argsDirect);
			UtilsCL.setKernelArguments<T>(kernelDirectSequentialSum, argsDirectSeqSum);
			UtilsCL.setKernelArguments<T>(kernelDirectCascadeSum, argsDirectCascadeSum);
			UtilsCL.setKernelArguments<T>(kernelDirectCascadeSumBlocks, argsDirectCascadeSumBlocks);
			UtilsCL.setKernelArguments<T>(kernelPreReverse, argsPreReverse);
			UtilsCL.setKernelArguments<T>(kernelReverse, argsReverse);
			UtilsCL.setKernelArguments<T>(kernelReverseSequentialSum, argsReverseSeqSum);
			UtilsCL.setKernelArguments<T>(kernelReverseCascadeSum, argsReverseCascadeSum);
			UtilsCL.setKernelArguments<T>(kernelReverseCascadeSumBlocks, argsReverseCascadeSumBlocks);
		}

		override public T doIteration(int iter)
		{//Q = 9.5*N2*N1*Log(N1,2) - 8*N2*N1;N2 can be == N1
			Func<int, int, T>[] funcs = { alphaCoeff , diagElem };
			BufferOCL<T>[] arOCL = { alphaCoefs , diagonalElements };
			void calculate(int idx)
			{
				T[] ar = new T[N1 - 1];
				fillArray(ar, funcs[idx]);
				commands.WriteToBuffer(ar, arOCL[idx], true, 0, 0, ar.Length, null);
				ar = null;
			}
			Parallel.For(0, 2, new ParallelOptions() { MaxDegreeOfParallelism = 2 }, i => { calculate(i); });

			gWorkSize[0] = N1 - 1;
			commands.Execute(kernelAlfa, null, gWorkSize, null, null);
			commands.Finish();

			int thresholdCascadeSum = 256;//int.MaxValue
			int maxThreadsCascadeSum = (int)kernelReverseCascadeSum.GetWorkGroupSize(commands.Device);

			for (int k = 1; k <= n - 1; k++)
			{
				int m = 1 << (k - 1);
				int t = m << 1;

				gWorkSize[0] = (N1 / t - 1) *  m;
				kernelDirect.SetValueArgument(0, t);
				kernelDirect.SetValueArgument(1, m);
				commands.Execute(kernelDirect, null, gWorkSize, null, null);
				commands.Finish();

				if (m > 1)
				{
					if (m >= thresholdCascadeSum)
					{
						gWorkSize[0] = m;
						kernelDirectCascadeSum.SetValueArgument(1, t);
						kernelDirectCascadeSum.SetValueArgument(2, m);
						int nBlocks = m / maxThreadsCascadeSum;
						kernelDirectCascadeSum.SetValueArgument(6, nBlocks);
						int threadsPerBlock = int.Min(m, maxThreadsCascadeSum);
						lWorkSize[0] = threadsPerBlock;
						for (int i = 0; i < N1 / t - 1; i++)
						{
							kernelDirectCascadeSum.SetValueArgument(0, i);
							commands.Execute(kernelDirectCascadeSum, null, gWorkSize, lWorkSize, null);
							commands.Finish();

							if (nBlocks > 1)
							{
								kernelDirectCascadeSumBlocks.SetValueArgument(0, i);
								kernelDirectCascadeSumBlocks.SetValueArgument(1, t);
								kernelDirectCascadeSumBlocks.SetValueArgument(2, m);
								kernelDirectCascadeSumBlocks.SetValueArgument(5, nBlocks);
								kernelDirectCascadeSumBlocks.SetValueArgument(6, threadsPerBlock);
								commands.Execute(kernelDirectCascadeSumBlocks, null, gWorkSizeSumBlocks, null, null);
								commands.Finish();
							}
						}
					}
					else
					{
						gWorkSize[0] = (N1 / t - 1);
						kernelDirectSequentialSum.SetValueArgument(0, t);
						kernelDirectSequentialSum.SetValueArgument(1, m);
						commands.Execute(kernelDirectSequentialSum, null, gWorkSize, null, null);
						commands.Finish();
					}
				}

				showProgress();
				if (areIterationsCanceled()) return T.Zero;
			}

			for (int k = n; k >= 1; k--)// reverse steps
			{
				int m = 1 << (k - 1);
				int t = m << 1;

				gWorkSize[0] = N1 / t;
				kernelPreReverse.SetValueArgument(0, t);
				kernelPreReverse.SetValueArgument(1, m);
				commands.Execute(kernelPreReverse, null, gWorkSize, null, null);
				commands.Finish();

				gWorkSize[0] = N1 / 2;
				kernelReverse.SetValueArgument(0, t);
				kernelReverse.SetValueArgument(1, m);
				commands.Execute(kernelReverse, null, gWorkSize, null, null);
				commands.Finish();

				if (m > 1)
				{
					if (m >= thresholdCascadeSum)
					{
						gWorkSize[0] = m;
						kernelReverseCascadeSum.SetValueArgument(1, t);
						kernelReverseCascadeSum.SetValueArgument(2, m);
						int nBlocks = m / maxThreadsCascadeSum;
						kernelReverseCascadeSum.SetValueArgument(6, nBlocks);
						int threadsPerBlock = int.Min(m, maxThreadsCascadeSum);
						lWorkSize[0] = threadsPerBlock;
						for (int i = 0; i < N1 / t; i++)
						{
							kernelReverseCascadeSum.SetValueArgument(0, i);
							commands.Execute(kernelReverseCascadeSum, null, gWorkSize, lWorkSize, null);
							commands.Finish();

							if (nBlocks > 1)
							{
								kernelReverseCascadeSumBlocks.SetValueArgument(0, i);
								kernelReverseCascadeSumBlocks.SetValueArgument(1, t);
								kernelReverseCascadeSumBlocks.SetValueArgument(2, m);
								kernelReverseCascadeSumBlocks.SetValueArgument(5, nBlocks);
								kernelReverseCascadeSumBlocks.SetValueArgument(6, threadsPerBlock);
								commands.Execute(kernelReverseCascadeSumBlocks, null, gWorkSizeSumBlocks, null, null);
								commands.Finish();
							}
						}
					}
					else
					{
						gWorkSize[0] = N1 / t;
						kernelReverseSequentialSum.SetValueArgument(0, t);
						kernelReverseSequentialSum.SetValueArgument(1, m);
						commands.Execute(kernelReverseSequentialSum, null, gWorkSize, null, null);
						commands.Finish();
					}
				}

				if (lstBitmap != null && k != 1) addPicture(m, N1 - m, t);//if k == 1, then picture will be the same as final(which is added in external code)

				showProgress();
				if (areIterationsCanceled()) return T.Zero;
			}

			deviceBufferToHostBuffer();

			return T.Zero;//epsilon, == 0 because no more iterations(only one iteration - direct(not iteration) method)
		}

		T alphaCoeff(int k,int l)
		{
			int m = 1 << (k - 1);
			T sin = T.Sin(T.Pi * T.CreateTruncating(2 * l - 1) / T.CreateTruncating(1 << k)) / T.CreateTruncating(m);
			if (l % 2 == 0) sin = -sin;
			return sin;
		}

		override public void cleanup()
		{
			UtilsCL.disposeBuf(ref alphaCoefs);
			UtilsCL.disposeBuf(ref alfa);
			UtilsCL.disposeBuf(ref ac);
			UtilsCL.disposeBuf(ref vv);
			UtilsCL.disposeBuf(ref diagonalElements);
			UtilsCL.disposeBuf(ref sum);
			UtilsCL.disposeBuf(ref alfaOffsetsOCL);
			UtilsCL.disposeBuf(ref alfaCountsOCL);

			UtilsCL.disposeKP(ref kernelAlfa);
			UtilsCL.disposeKP(ref kernelDirect);
			UtilsCL.disposeKP(ref kernelReverse);
			UtilsCL.disposeKP(ref kernelDirectSequentialSum);
			UtilsCL.disposeKP(ref kernelDirectCascadeSum);
			UtilsCL.disposeKP(ref kernelDirectCascadeSumBlocks);
			UtilsCL.disposeKP(ref kernelPreReverse);
			UtilsCL.disposeKP(ref kernelReverseSequentialSum);
			UtilsCL.disposeKP(ref kernelReverseCascadeSum);
			UtilsCL.disposeKP(ref kernelReverseCascadeSumBlocks);

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
					int kUp = αCC.upperBound(diagElem(k, l), alfaVectorLength - 1);
					//int kUp = alfaVectorLength - 1;//for test
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

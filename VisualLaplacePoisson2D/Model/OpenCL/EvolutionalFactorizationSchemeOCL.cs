using Cloo;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using VLP2D.Common;

namespace VLP2D.Model
{
	class EvolutionalFactorizationSchemeOCL<T> : Iterative1DScheme<T>, IScheme<T> where T : unmanaged, INumber<T>, IRootFunctions<T>, IPowerFunctions<T>, IMinMaxValue<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>, ITrigonometricFunctions<T>, ILogarithmicFunctions<T>
	{
		T stepX2, stepY2, eps, hx2, hy2, xMax, yMax, ratioOfSquaresOfSteps;
		T[] alphaX, alphaY;//beta is placed to result
		int cXSegments, cYSegments;

		CommandQueueOCL commands;
		KernelOCL[] kernels;
		BufferOCL<T> unOCL0, unOCL1, unOCLm, fn;
		BufferOCL<T> alphaXOCL, alphaYOCL;
		T[] un;
		long[] gWork1DOffset = new long[] { 1 };
		long[] gWork2DOffset = new long[] { 1, 1 };
		long[] workSizeX, workSizeY, workSize2DInternalPoints;
		bool uuChanged = false;
		static T _2 = T.CreateTruncating(2), _4 = T.CreateTruncating(4);
		T[] τ;

		public EvolutionalFactorizationSchemeOCL(RectangleData<T> rectData, Func<T, T, T> fKsi, T epsIn, PlatformOCL platform, DeviceOCL device) :
			base(rectData.xMin, rectData.yMin, rectData.stepX, rectData.stepY)
		{
			UtilsCL.checkDeviceSupportDouble<T>(device);
			stepX2 = stepX * stepX;
			stepY2 = stepY * stepY;
			cXSegments = rectData.cXSegments;
			cYSegments = rectData.cYSegments;
			xMax = rectData.xMax;
			yMax = rectData.yMax;

			dimX = cXSegments + 1;
			dimY = cYSegments + 1;

			un = new T[dimX * dimY];

			alphaX = new T[cXSegments];
			alphaY = new T[cYSegments];

			eps = epsIn;

			commands = UtilsCL.createCommandQueue(platform, device, CommandQueueFlagsOCL.None);

			unOCL0 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dimX * dimY);
			unOCL1 = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dimX * dimY);
			unOCLm = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, dimX * dimY);

			alphaXOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, alphaX.Length);
			alphaYOCL = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadWrite, alphaY.Length);

			if (fKsi != null)
			{
				GridIterator.iterate(dimX - 1, dimY - 1, (i, j) => un[i * dimY + j] = fKsi(xMin + stepX * T.CreateTruncating(i), yMin + stepY * T.CreateTruncating(j)));
				fn = new BufferOCL<T>(commands.Context, MemoryFlagsOCL.ReadOnly | MemoryFlagsOCL.CopyHostPointer, un);
			}

			ratioOfSquaresOfSteps = stepX2 / stepY2;
			hx2 = stepX2 / _4;//== (stepX/2)^2
			hy2 = stepY2 / _4;//== (stepY/2)^2

			τ = UtilsLT<T>.calculateTau(eps, hx2, hy2, xMax - xMin, yMax - yMin);

			kernels = new KernelOCL[2];

			workSizeX = new long[] { dimX - 2 };
			workSizeY = new long[] { dimY - 2 };
			workSize2DInternalPoints = new long[] { workSizeX[0], workSizeY[0] };

			kernels[0] = EvolutionalFactorizationProgramsOCL.createProgramProgonkaX<T>(dimX, dimY, fn != null, commands.Context, commands.Device);
			initKernel0Arguments();

			kernels[1] = EvolutionalFactorizationProgramsOCL.createProgramProgonkaY<T>(dimY, commands.Context, commands.Device);
			initKernel1Arguments();
		}

		public T doIteration(int iter)
		{
			uuChanged = true;

			evolutionalFactorization(τ[iter]);

			UtilsSwap.swap(ref unOCL0, ref unOCL1);

			return T.One;
		}

		void evolutionalFactorization(T τ)
		{
			T twoDivTau = _2 / τ;

			calcAlpha(_2 + twoDivTau * hx2, _2 + twoDivTau * hy2);

			setKernel0Arguments(twoDivTau);
			commands.Execute(kernels[0], gWork1DOffset, workSizeY, null, null);
			commands.Finish();

			setKernel1Arguments(twoDivTau, τ);
			commands.Execute(kernels[1], gWork1DOffset, workSizeX, null, null);
			commands.Finish();
		}

		void initKernel0Arguments()
		{
			kernels[0].SetMemoryArgument(0, unOCL0);
			kernels[0].SetMemoryArgument(1, unOCLm);
			kernels[0].SetMemoryArgument(2, alphaXOCL);
			kernels[0].SetValueArgument(3, stepX2);
			kernels[0].SetValueArgument(4, T.Zero);//instead of twoDivTau
			kernels[0].SetValueArgument(5, ratioOfSquaresOfSteps);
			if (fn != null) kernels[0].SetMemoryArgument(6, fn);
		}

		void initKernel1Arguments()
		{
			kernels[1].SetMemoryArgument(0, unOCLm);
			kernels[1].SetMemoryArgument(1, unOCL1);
			kernels[1].SetMemoryArgument(2, unOCL0);
			kernels[1].SetMemoryArgument(3, alphaYOCL);
			kernels[1].SetValueArgument(4, hy2);
			kernels[1].SetValueArgument(5, T.Zero);//instead of twoDivTau
			kernels[1].SetValueArgument(6, T.Zero);//instead of τ
		}

		void setKernel0Arguments(T twoDivTau)
		{
			kernels[0].SetMemoryArgument(0, unOCL0);
			kernels[0].SetValueArgument(4, twoDivTau);
		}

		void setKernel1Arguments(T twoDivTau, T τ)
		{
			kernels[1].SetMemoryArgument(1, unOCL1);
			kernels[1].SetMemoryArgument(2, unOCL0);
			kernels[1].SetValueArgument(5, twoDivTau);
			kernels[1].SetValueArgument(6, τ);
		}

		void calcAlpha(T bx, T by)
		{
			Action<T, T[], int> calc = (diag, alpha, bound) =>
			{
				alpha[0] = T.Zero;
				for (int i = 1; i <= bound; i++) alpha[i] = T.One / (diag - alpha[i - 1]);
			};

			Action[] actions = [() => calc(bx, alphaX, cXSegments - 1), () => calc(by, alphaY, cYSegments - 1)];
			Parallel.For(0, 2, GridIterator.optionsParallel, j => actions[j].Invoke());

			commands.WriteToBuffer(alphaX, alphaXOCL, true, null);
			commands.WriteToBuffer(alphaY, alphaYOCL, true, null);
		}

		public override T[] getArray()
		{
			if (uuChanged)
			{
				uuChanged = false;

				commands.ReadFromBuffer(unOCL0, ref un, true, null);
			}

			return un;
		}

		public virtual void initAfterBoundariesAndInitialIterationInited()
		{
			commands.WriteToBuffer(un, unOCL0, true, null);
			commands.CopyBuffer(unOCL0, unOCL1, null);
		}

		public virtual int maxIterations() { return τ.Length; }
		public bool shouldReportProgress() { return true; }
		public void cancelIterations() { }
		public override IterationsKind iterationsKind() => IterationsKind.knownInAdvance;

		public virtual void cleanup()
		{
			UtilsCL.disposeQC(ref commands);
			UtilsCL.disposeKP(ref kernels[0]);
			UtilsCL.disposeKP(ref kernels[1]);
			UtilsCL.disposeBuf(ref unOCL0);
			UtilsCL.disposeBuf(ref unOCL1);
			UtilsCL.disposeBuf(ref unOCLm);
			UtilsCL.disposeBuf(ref fn);
			UtilsCL.disposeBuf(ref alphaXOCL);
			UtilsCL.disposeBuf(ref alphaYOCL);
		}
	}
}

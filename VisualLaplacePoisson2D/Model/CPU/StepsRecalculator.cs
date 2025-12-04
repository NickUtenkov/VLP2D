#define UsePower2//for Lomont FFT

using ELW.Library.Math.Tools;
using System.Numerics;
using VLP2D.Common;

namespace VLP2D.Model
{
	internal class StepsRecalculator<T> where T : unmanaged, INumber<T>, ITrigonometricFunctions<T>, IPowerFunctions<T>, IExponentialFunctions<T>, IHyperbolicFunctions<T>, ILogarithmicFunctions<T>, IRootFunctions<T>, IMinMaxValue<T>
	{
		public T stepX, stepY;
		public int cXSegments, cYSegments;

		public StepsRecalculator()
		{
		}

		public void recalculateSteps(string strXMax, int cXSegmentsIn, string strYMax, int cYSegmentsIn, PlatformAndSchemeIndex platformScheme, bool isVarSepProgonkaGPU, VarSepMethodsEnum varSepMethodCPU)
		{
			this.cXSegments = cXSegmentsIn;
			this.cYSegments = cYSegmentsIn;

			if (platformScheme.platrofm == PlatformEnum.CPU)
			{
				SchemeCPUEnum idxScheme = (SchemeCPUEnum)platformScheme.idxScheme;
				if (idxScheme == SchemeCPUEnum.MultiGrid)//make pow 2
				{
					cXSegments = 1 << Utils.calculatePowOf2(cXSegments);
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);
				}
				else if (idxScheme == SchemeCPUEnum.SOR || idxScheme == SchemeCPUEnum.SlidingIteration)//work group equal sizes
				{
					if (cXSegments % 2 == 0) cXSegments++;
					if (cYSegments % 2 == 0) cYSegments++;
				}
				else if (idxScheme == SchemeCPUEnum.CompleteReduction)//make cXSegments pow 2
				{
					T ratio = T.CreateTruncating(cYSegments) / T.CreateTruncating(cXSegments);
					cXSegments = 1 << Utils.calculatePowOf2(cXSegments);

					cYSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cXSegments));
					if ((cYSegments & 1) == 0) cYSegments++;//make cYSegments odd for meeting progonka(used only one formula in such a case),MeetingProgonka
				}
				else if (idxScheme == SchemeCPUEnum.VariablesSeparation)
				{
					T ratio = T.CreateTruncating(cXSegments) / T.CreateTruncating(cYSegments);
#if UsePower2
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);
#else
					//if ((cYSegments & 1) == 1) cYSegments++;//make cYSegments even for FFTSineTransform correct work
#endif
					if (varSepMethodCPU == VarSepMethodsEnum.Progonka)//can not be pow of 2(because fft for cXSegments is not used)
					{
						cXSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cYSegments));
						if ((cXSegments & 1) == 0) cXSegments++;//make cXSegments odd for meeting progonka(used only one formula in such a case),MeetingProgonka
					}
					else if (varSepMethodCPU == VarSepMethodsEnum.Reduction)
					{
						cXSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cYSegments));
					}
					else if (varSepMethodCPU == VarSepMethodsEnum.Marching)
					{
						cXSegments = (1 << Utils.calculatePowOf2(cXSegments - 1)) + 1;//make Nx = 2kL + 1;SNE p.231, (25)
					}
					else//VarSepMethodsEnum.FFT
					{
#if UsePower2
						cXSegments = 1 << Utils.calculatePowOf2(cXSegments);
#else
						//if ((cXSegments & 1) == 1) cXSegments++;//make cXSegments even for FFTSineTransform correct work
#endif
					}
				}
				else if (idxScheme == SchemeCPUEnum.IncompleteReduction)
				{//differs from OCL variant
				 //make cXSegments pow 2;cYSegments can not be pow of 2(because fft for cYSegments is not used)
					T ratio = T.CreateTruncating(cYSegments) / T.CreateTruncating(cXSegments);
					cXSegments = 1 << Utils.calculatePowOf2(cXSegments);

					cYSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cXSegments));
				}
				else if (idxScheme == SchemeCPUEnum.Marching)
				{
					T ratio = T.CreateTruncating(cYSegments) / T.CreateTruncating(cXSegments);
					cXSegments = (1 << Utils.calculatePowOf2(cXSegments - 1)) + 1;//make Nx = 2kL + 1;SNE p.231, (25)

#if UsePower2
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);//make Ny == 2^x for FFT
#else
					cYSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cXSegments));//FFTW support any powers for size of FFT
					//if ((cYSegments & 1) == 1) cYSegments++;//make cXSegments even for FFTSineTransform correct work
#endif
				}
			}
			else if (platformScheme.platrofm == PlatformEnum.OCL)
			{
				SchemeOCLEnum idxScheme = (SchemeOCLEnum)platformScheme.idxScheme;
				if (idxScheme == SchemeOCLEnum.MultiGridOCL)//make pow 2
				{
					cXSegments = 1 << Utils.calculatePowOf2(cXSegments);
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);
				}
				else if (idxScheme == SchemeOCLEnum.SOROCL || idxScheme == SchemeOCLEnum.SlidingIterationOCL)//work group equal sizes
				{
					if (cXSegments % 2 == 0) cXSegments++;
					if (cYSegments % 2 == 0) cYSegments++;
				}
				else if (idxScheme == SchemeOCLEnum.CompleteReductionOCL)//make cXSegments pow 2
				{
					T ratio = T.CreateTruncating(cYSegments) / T.CreateTruncating(cXSegments);
					cXSegments = 1 << Utils.calculatePowOf2(cXSegments);

					cYSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cXSegments));
					if ((cYSegments & 1) == 0) cYSegments++;//make cYSegments odd for meeting progonka(used only one formula in such a case),MeetingProgonka
				}
				else if (idxScheme == SchemeOCLEnum.VariablesSeparationOCL)//make cYSegments(not cXSegments!) pow 2
				{
					T ratio = T.CreateTruncating(cXSegments) / T.CreateTruncating(cYSegments);
#if UsePower2
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);//make cYSegments(not cXSegments!) pow 2
#else
					//if ((cYSegments & 1) == 1) cYSegments++;//make cXSegments even for FFTSineTransform correct work
#endif
					if (isVarSepProgonkaGPU)//can not be pow of 2(because fft for cXSegments is not used)
					{
						cXSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cYSegments));
						if ((cXSegments & 1) == 0) cXSegments++;//make cXSegments odd for meeting progonka(used only one formula in such a case),MeetingProgonka
					}
					else
					{
#if UsePower2
						cXSegments = 1 << Utils.calculatePowOf2(cXSegments);
#else
						//if ((cXSegments & 1) == 1) cXSegments++;//make cXSegments even for FFTSineTransform correct work
#endif
					}
				}
				else if (idxScheme == SchemeOCLEnum.IncompleteReductionOCL)
				{//differs from CPU variant
				 //make cYSegments pow 2;cXSegments can not be pow of 2(because fft for cXSegments is not used) - as in VariablesSeparation progonka method
					T ratio = T.CreateTruncating(cXSegments) / T.CreateTruncating(cYSegments);
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);

					cXSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cYSegments));
					if ((cXSegments & 1) == 0) cXSegments++;//make cXSegments odd for meeting progonka(used only one formula in such a case),MeetingProgonka
				}
				else if (idxScheme == SchemeOCLEnum.MarchingOCL)
				{
					cXSegments = (1 << Utils.calculatePowOf2(cXSegments - 1)) + 1;//make Nx = 2kL + 1;SNE p.231, (25)
#if UsePower2
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);//make Ny == 2^x for FFT
#else
					/*if (!Utils.containsDividers(cYSegments, new int[] { 2, 3, 5, 7, 11, 13 }))//clFFT supports only such powers
					{
						cYSegments = 1 << Utils.calculatePowOf2(cYSegments);//make Ny == 2^x for FFT
					}*/
#endif
				}
			}
			else if (platformScheme.platrofm == PlatformEnum.CUDA)
			{
				SchemeCUDAEnum idxScheme = (SchemeCUDAEnum)platformScheme.idxScheme;
				if (idxScheme == SchemeCUDAEnum.SORCUDA || idxScheme == SchemeCUDAEnum.SlidingIterationCUDA)//work group equal sizes
				{
					if (cXSegments % 2 == 0) cXSegments++;
					if (cYSegments % 2 == 0) cYSegments++;
				}
				else if (idxScheme == SchemeCUDAEnum.MultiGridCUDA)//make pow 2
				{
					cXSegments = 1 << Utils.calculatePowOf2(cXSegments);
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);
				}
				else if (idxScheme == SchemeCUDAEnum.CompleteReductionCUDA)//make cXSegments pow 2
				{
					T ratio = T.CreateTruncating(cYSegments) / T.CreateTruncating(cXSegments);
					cXSegments = 1 << Utils.calculatePowOf2(cXSegments);

					cYSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cXSegments));
					if ((cYSegments & 1) == 0) cYSegments++;//make cYSegments odd for meeting progonka(used only one formula in such a case),MeetingProgonka
				}
				else if (idxScheme == SchemeCUDAEnum.VariablesSeparationCUDA)
				{
					T ratio = T.CreateTruncating(cXSegments) / T.CreateTruncating(cYSegments);
#if UsePower2
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);//make cYSegments(not cXSegments!) pow 2
#else
					//if ((cYSegments & 1) == 1) cYSegments++;//make cXSegments even for FFTSineTransform correct work
#endif
					if (isVarSepProgonkaGPU)//can not be pow of 2(because fft for cXSegments is not used)
					{
						cXSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cYSegments));
						if ((cXSegments & 1) == 0) cXSegments++;//make cXSegments odd for meeting progonka(used only one formula in such a case),MeetingProgonka
					}
					else
					{
#if UsePower2
						cXSegments = 1 << Utils.calculatePowOf2(cXSegments);
#else
						//if ((cXSegments & 1) == 1) cXSegments++;//make cXSegments even for FFTSineTransform correct work
#endif
					}
				}
				else if (idxScheme == SchemeCUDAEnum.FACRCUDA)
				{//differs from CPU variant
				 //make cYSegments pow 2;cXSegments can not be pow of 2(because fft for cXSegments is not used) - as in VariablesSeparation progonka method
					T ratio = T.CreateTruncating(cXSegments) / T.CreateTruncating(cYSegments);
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);

					cXSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cYSegments));
					if ((cXSegments & 1) == 0) cXSegments++;//make cXSegments odd for meeting progonka(used only one formula in such a case),MeetingProgonka
				}
				else if (idxScheme == SchemeCUDAEnum.MarchingCUDA)
				{
					T ratio = T.CreateTruncating(cYSegments) / T.CreateTruncating(cXSegments);
					cXSegments = (1 << Utils.calculatePowOf2(cXSegments - 1)) + 1;//make Nx = 2kL + 1;SNE p.231, (25)

#if UsePower2
					cYSegments = 1 << Utils.calculatePowOf2(cYSegments);//make Ny == 2^x for FFT
#else
					cYSegments = (int)uint.CreateTruncating(ratio * T.CreateTruncating(cXSegments));//cuFFT support any powers for size of FFT
					//if ((cYSegments & 1) == 1) cYSegments++;//make cXSegments even for FFTSineTransform correct work
#endif
				}
			}

			Utils.addCustomFunctions<T>();
			Calculator0D<T> calculator = new Calculator0D<T>();

			calculator.setExpression(strXMax);
			stepX = calculator.Calculate() / T.CreateTruncating(cXSegments);

			calculator.setExpression(strYMax);
			stepY = calculator.Calculate() / T.CreateTruncating(cYSegments);
		}
	}
}

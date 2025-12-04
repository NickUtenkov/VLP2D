#region License

/*

Copyright (c) 2009 - 2011 Fatjon Sakiqi

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

*/

#endregion

namespace Cloo
{
    using System;
    using System.Diagnostics;

    /// <summary>
    /// Represents an error state that occurred while executing an OpenCL API call.
    /// </summary>
    /// <seealso cref="ErrorCode"/>
    public class ExceptionOCL : ApplicationException
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ErrorCodeOCL code;

        #endregion

        #region Properties

        /// <summary>
        /// Gets the <see cref="ErrorCode"/> of the <see cref="ExceptionOCL"/>.
        /// </summary>
        public ErrorCodeOCL ErrorCode { get { return code; } }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new <see cref="ExceptionOCL"/> with a specified <see cref="ErrorCode"/>.
        /// </summary>
        /// <param name="code"> A <see cref="ErrorCode"/>. </param>
        public ExceptionOCL(ErrorCodeOCL code)
            : base("OpenCL error code detected: " + code.ToString() + ".")
        {
            this.code = code;
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Checks for an OpenCL error code and throws a <see cref="ExceptionOCL"/> if such is encountered.
        /// </summary>
        /// <param name="errorCode"> The value to be checked for an OpenCL error. </param>
        public static void ThrowOnError(int errorCode)
        {
            ThrowOnError((ErrorCodeOCL)errorCode);
        }

        /// <summary>
        /// Checks for an OpenCL error code and throws a <see cref="ExceptionOCL"/> if such is encountered.
        /// </summary>
        /// <param name="errorCode"> The OpenCL error code. </param>
        public static void ThrowOnError(ErrorCodeOCL errorCode)
        {
            switch (errorCode)
            {
                case ErrorCodeOCL.Success:
                    return;

                case ErrorCodeOCL.DeviceNotFound:
                    throw new DeviceNotFoundException();

                case ErrorCodeOCL.DeviceNotAvailable:
                    throw new DeviceNotAvailableException();

                case ErrorCodeOCL.CompilerNotAvailable:
                    throw new CompilerNotAvailableException();

                case ErrorCodeOCL.MemoryObjectAllocationFailure:
                    throw new MemoryObjectAllocationFailureException();

                case ErrorCodeOCL.OutOfResources:
                    throw new OutOfResourcesException();

                case ErrorCodeOCL.OutOfHostMemory:
                    throw new OutOfHostMemoryException();

                case ErrorCodeOCL.ProfilingInfoNotAvailable:
                    throw new ProfilingInfoNotAvailableException();

                case ErrorCodeOCL.MemoryCopyOverlap:
                    throw new MemoryCopyOverlapException();

                case ErrorCodeOCL.ImageFormatMismatch:
                    throw new ImageFormatMismatchException();

                case ErrorCodeOCL.ImageFormatNotSupported:
                    throw new ImageFormatNotSupportedException();

                case ErrorCodeOCL.BuildProgramFailure:
                    throw new BuildProgramFailureException();

                case ErrorCodeOCL.MapFailure:
                    throw new MapFailureException();

                case ErrorCodeOCL.InvalidValue:
                    throw new InvalidValueException();

                case ErrorCodeOCL.InvalidDeviceType:
                    throw new InvalidDeviceTypeException();

                case ErrorCodeOCL.InvalidPlatform:
                    throw new InvalidPlatformException();

                case ErrorCodeOCL.InvalidDevice:
                    throw new InvalidDeviceException();

                case ErrorCodeOCL.InvalidContext:
                    throw new InvalidContextException();

                case ErrorCodeOCL.InvalidCommandQueueFlags:
                    throw new InvalidCommandQueueFlagsException();

                case ErrorCodeOCL.InvalidCommandQueue:
                    throw new InvalidCommandQueueException();

                case ErrorCodeOCL.InvalidHostPointer:
                    throw new InvalidHostPointerException();

                case ErrorCodeOCL.InvalidMemoryObject:
                    throw new InvalidMemoryObjectException();

                case ErrorCodeOCL.InvalidImageFormatDescriptor:
                    throw new InvalidImageFormatDescriptorException();

                case ErrorCodeOCL.InvalidImageSize:
                    throw new InvalidImageSizeException();

                case ErrorCodeOCL.InvalidSampler:
                    throw new InvalidSamplerException();

                case ErrorCodeOCL.InvalidBinary:
                    throw new InvalidBinaryException();

                case ErrorCodeOCL.InvalidBuildOptions:
                    throw new InvalidBuildOptionsException();

                case ErrorCodeOCL.InvalidProgram:
                    throw new InvalidProgramException();

                case ErrorCodeOCL.InvalidProgramExecutable:
                    throw new InvalidProgramExecutableException();

                case ErrorCodeOCL.InvalidKernelName:
                    throw new InvalidKernelNameException();

                case ErrorCodeOCL.InvalidKernelDefinition:
                    throw new InvalidKernelDefinitionException();

                case ErrorCodeOCL.InvalidKernel:
                    throw new InvalidKernelException();

                case ErrorCodeOCL.InvalidArgumentIndex:
                    throw new InvalidArgumentIndexException();

                case ErrorCodeOCL.InvalidArgumentValue:
                    throw new InvalidArgumentValueException();

                case ErrorCodeOCL.InvalidArgumentSize:
                    throw new InvalidArgumentSizeException();

                case ErrorCodeOCL.InvalidKernelArguments:
                    throw new InvalidKernelArgumentsException();

                case ErrorCodeOCL.InvalidWorkDimension:
                    throw new InvalidWorkDimensionsException();

                case ErrorCodeOCL.InvalidWorkGroupSize:
                    throw new InvalidWorkGroupSizeException();

                case ErrorCodeOCL.InvalidWorkItemSize:
                    throw new InvalidWorkItemSizeException();

                case ErrorCodeOCL.InvalidGlobalOffset:
                    throw new InvalidGlobalOffsetException();

                case ErrorCodeOCL.InvalidEventWaitList:
                    throw new InvalidEventWaitListException();

                case ErrorCodeOCL.InvalidEvent:
                    throw new InvalidEventException();

                case ErrorCodeOCL.InvalidOperation:
                    throw new InvalidOperationException();

                case ErrorCodeOCL.InvalidGLObject:
                    throw new InvalidGLObjectException();

                case ErrorCodeOCL.InvalidBufferSize:
                    throw new InvalidBufferSizeException();

                case ErrorCodeOCL.InvalidMipLevel:
                    throw new InvalidMipLevelException();

                default:
                    throw new ExceptionOCL(errorCode);
            }
        }

        #endregion
    }

    #region Exception classes

    // Disable CS1591 warnings (missing XML comment for publicly visible type or member).
    #pragma warning disable 1591

    public class DeviceNotFoundException : ExceptionOCL
    { public DeviceNotFoundException() : base(ErrorCodeOCL.DeviceNotFound) { } }

    public class DeviceNotAvailableException : ExceptionOCL
    { public DeviceNotAvailableException() : base(ErrorCodeOCL.DeviceNotAvailable) { } }

    public class CompilerNotAvailableException : ExceptionOCL
    { public CompilerNotAvailableException() : base(ErrorCodeOCL.CompilerNotAvailable) { } }

    public class MemoryObjectAllocationFailureException : ExceptionOCL
    { public MemoryObjectAllocationFailureException() : base(ErrorCodeOCL.MemoryObjectAllocationFailure) { } }

    public class OutOfResourcesException : ExceptionOCL
    { public OutOfResourcesException() : base(ErrorCodeOCL.OutOfResources) { } }

    public class OutOfHostMemoryException : ExceptionOCL
    { public OutOfHostMemoryException() : base(ErrorCodeOCL.OutOfHostMemory) { } }

    public class ProfilingInfoNotAvailableException : ExceptionOCL
    { public ProfilingInfoNotAvailableException() : base(ErrorCodeOCL.ProfilingInfoNotAvailable) { } }

    public class MemoryCopyOverlapException : ExceptionOCL
    { public MemoryCopyOverlapException() : base(ErrorCodeOCL.MemoryCopyOverlap) { } }

    public class ImageFormatMismatchException : ExceptionOCL
    { public ImageFormatMismatchException() : base(ErrorCodeOCL.ImageFormatMismatch) { } }

    public class ImageFormatNotSupportedException : ExceptionOCL
    { public ImageFormatNotSupportedException() : base(ErrorCodeOCL.ImageFormatNotSupported) { } }

    public class BuildProgramFailureException : ExceptionOCL
    { public BuildProgramFailureException() : base(ErrorCodeOCL.BuildProgramFailure) { } }

    public class MapFailureException : ExceptionOCL
    { public MapFailureException() : base(ErrorCodeOCL.MapFailure) { } }

    public class InvalidValueException : ExceptionOCL
    { public InvalidValueException() : base(ErrorCodeOCL.InvalidValue) { } }

    public class InvalidDeviceTypeException : ExceptionOCL
    { public InvalidDeviceTypeException() : base(ErrorCodeOCL.InvalidDeviceType) { } }

    public class InvalidPlatformException : ExceptionOCL
    { public InvalidPlatformException() : base(ErrorCodeOCL.InvalidPlatform) { } }

    public class InvalidDeviceException : ExceptionOCL
    { public InvalidDeviceException() : base(ErrorCodeOCL.InvalidDevice) { } }

    public class InvalidContextException : ExceptionOCL
    { public InvalidContextException() : base(ErrorCodeOCL.InvalidContext) { } }

    public class InvalidCommandQueueFlagsException : ExceptionOCL
    { public InvalidCommandQueueFlagsException() : base(ErrorCodeOCL.InvalidCommandQueueFlags) { } }

    public class InvalidCommandQueueException : ExceptionOCL
    { public InvalidCommandQueueException() : base(ErrorCodeOCL.InvalidCommandQueue) { } }

    public class InvalidHostPointerException : ExceptionOCL
    { public InvalidHostPointerException() : base(ErrorCodeOCL.InvalidHostPointer) { } }

    public class InvalidMemoryObjectException : ExceptionOCL
    { public InvalidMemoryObjectException() : base(ErrorCodeOCL.InvalidMemoryObject) { } }

    public class InvalidImageFormatDescriptorException : ExceptionOCL
    { public InvalidImageFormatDescriptorException() : base(ErrorCodeOCL.InvalidImageFormatDescriptor) { } }

    public class InvalidImageSizeException : ExceptionOCL
    { public InvalidImageSizeException() : base(ErrorCodeOCL.InvalidImageSize) { } }

    public class InvalidSamplerException : ExceptionOCL
    { public InvalidSamplerException() : base(ErrorCodeOCL.InvalidSampler) { } }

    public class InvalidBinaryException : ExceptionOCL
    { public InvalidBinaryException() : base(ErrorCodeOCL.InvalidBinary) { } }

    public class InvalidBuildOptionsException : ExceptionOCL
    { public InvalidBuildOptionsException() : base(ErrorCodeOCL.InvalidBuildOptions) { } }

    public class InvalidProgramException : ExceptionOCL
    { public InvalidProgramException() : base(ErrorCodeOCL.InvalidProgram) { } }

    public class InvalidProgramExecutableException : ExceptionOCL
    { public InvalidProgramExecutableException() : base(ErrorCodeOCL.InvalidProgramExecutable) { } }

    public class InvalidKernelNameException : ExceptionOCL
    { public InvalidKernelNameException() : base(ErrorCodeOCL.InvalidKernelName) { } }

    public class InvalidKernelDefinitionException : ExceptionOCL
    { public InvalidKernelDefinitionException() : base(ErrorCodeOCL.InvalidKernelDefinition) { } }

    public class InvalidKernelException : ExceptionOCL
    { public InvalidKernelException() : base(ErrorCodeOCL.InvalidKernel) { } }

    public class InvalidArgumentIndexException : ExceptionOCL
    { public InvalidArgumentIndexException() : base(ErrorCodeOCL.InvalidArgumentIndex) { } }

    public class InvalidArgumentValueException : ExceptionOCL
    { public InvalidArgumentValueException() : base(ErrorCodeOCL.InvalidArgumentValue) { } }

    public class InvalidArgumentSizeException : ExceptionOCL
    { public InvalidArgumentSizeException() : base(ErrorCodeOCL.InvalidArgumentSize) { } }

    public class InvalidKernelArgumentsException : ExceptionOCL
    { public InvalidKernelArgumentsException() : base(ErrorCodeOCL.InvalidKernelArguments) { } }

    public class InvalidWorkDimensionsException : ExceptionOCL
    { public InvalidWorkDimensionsException() : base(ErrorCodeOCL.InvalidWorkDimension) { } }

    public class InvalidWorkGroupSizeException : ExceptionOCL
    { public InvalidWorkGroupSizeException() : base(ErrorCodeOCL.InvalidWorkGroupSize) { } }

    public class InvalidWorkItemSizeException : ExceptionOCL
    { public InvalidWorkItemSizeException() : base(ErrorCodeOCL.InvalidWorkItemSize) { } }

    public class InvalidGlobalOffsetException : ExceptionOCL
    { public InvalidGlobalOffsetException() : base(ErrorCodeOCL.InvalidGlobalOffset) { } }

    public class InvalidEventWaitListException : ExceptionOCL
    { public InvalidEventWaitListException() : base(ErrorCodeOCL.InvalidEventWaitList) { } }

    public class InvalidEventException : ExceptionOCL
    { public InvalidEventException() : base(ErrorCodeOCL.InvalidEvent) { } }

    public class InvalidOperationException : ExceptionOCL
    { public InvalidOperationException() : base(ErrorCodeOCL.InvalidOperation) { } }

    public class InvalidGLObjectException : ExceptionOCL
    { public InvalidGLObjectException() : base(ErrorCodeOCL.InvalidGLObject) { } }

    public class InvalidBufferSizeException : ExceptionOCL
    { public InvalidBufferSizeException() : base(ErrorCodeOCL.InvalidBufferSize) { } }

    public class InvalidMipLevelException : ExceptionOCL
    { public InvalidMipLevelException() : base(ErrorCodeOCL.InvalidMipLevel) { } }

    #endregion
}
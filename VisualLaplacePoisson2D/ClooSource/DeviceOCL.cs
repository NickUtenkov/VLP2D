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
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using Cloo.Bindings;

    /// <summary>
    /// Represents an OpenCL device.
    /// </summary>
    /// <value> A device is a collection of compute units. A command queue is used to queue commands to a device. Examples of commands include executing kernels, or reading and writing memory objects. OpenCL devices typically correspond to a GPU, a multi-core CPU, and other processors such as DSPs and the Cell/B.E. processor. </value>
    /// <seealso cref="CommandQueueOCL"/>
    /// <seealso cref="KernelOCL"/>
    /// <seealso cref="MemoryOCL"/>
    /// <seealso cref="PlatformOCL"/>
    public class DeviceOCL : ObjectOCL
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long addressBits;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool available;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool compilerAvailable;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string driverVersion;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool endianLittle;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool errorCorrectionSupport;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly DeviceExecutionCapabilitiesOCL executionCapabilities;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ReadOnlyCollection<string> extensions;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemoryCachelineSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemoryCacheSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly DeviceMemoryCacheTypeOCL globalMemoryCacheType;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long globalMemorySize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly bool imageSupport;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image2DMaxHeight;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image2DMaxWidth;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxDepth;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxHeight;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long image3DMaxWidth;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long localMemorySize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly DeviceLocalMemoryTypeOCL localMemoryType;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxClockFrequency;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxComputeUnits;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxConstantArguments;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxConstantBufferSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxMemAllocSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxParameterSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxReadImageArgs;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxSamplers;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWorkGroupSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWorkItemDimensions;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly ReadOnlyCollection<long> maxWorkItemSizes;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long maxWriteImageArgs;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long memBaseAddrAlign;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long minDataTypeAlignSize;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string name;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly PlatformOCL platform;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthChar;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthFloat;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthInt;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthLong;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long preferredVectorWidthShort;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string profile;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long profilingTimerResolution;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly CommandQueueFlagsOCL queueProperties;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly DeviceSingleCapabilitiesOCL singleCapabilities;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly DeviceTypesOCL type;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string vendor;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly long vendorId;
        [DebuggerBrowsable(DebuggerBrowsableState.Never)] private readonly string version;

        #endregion

        #region Properties

        /// <summary>
        /// The handle of the <see cref="DeviceOCL"/>.
        /// </summary>
        public CLDeviceHandle Handle
        {
            get;
            protected set;
        }

        /// <summary>
        /// Gets the default <see cref="DeviceOCL"/> address space size in bits.
        /// </summary>
        /// <value> Currently supported values are 32 or 64 bits. </value>
        public long AddressBits { get { return addressBits; } }

        /// <summary>
        /// Gets the availability state of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> Is <c>true</c> if the <see cref="DeviceOCL"/> is available and <c>false</c> otherwise. </value>
        public bool Available { get { return available; } }

        /// <summary>
        /// Gets the <see cref="CommandQueueFlagsOCL"/> supported by the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The <see cref="CommandQueueFlagsOCL"/> supported by the <see cref="DeviceOCL"/>. </value>
        public CommandQueueFlagsOCL CommandQueueFlags { get { return queueProperties; } }

        /// <summary>
        /// Gets the availability state of the OpenCL compiler of the <see cref="DeviceOCL.Platform"/>.
        /// </summary>
        /// <value> Is <c>true</c> if the implementation has a compiler available to compile the program source and <c>false</c> otherwise. This can be <c>false</c> for the embededed platform profile only. </value>
        public bool CompilerAvailable { get { return compilerAvailable; } }

        /// <summary>
        /// Gets the OpenCL software driver version string of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The version string in the form <c>major_number.minor_number</c>. </value>
        public string DriverVersion { get { return driverVersion; } }

        /// <summary>
        /// Gets the endianness of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> Is <c>true</c> if the <see cref="DeviceOCL"/> is a little endian device and <c>false</c> otherwise. </value>
        public bool EndianLittle { get { return endianLittle; } }

        /// <summary>
        /// Gets the error correction support state of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> Is <c>true</c> if the <see cref="DeviceOCL"/> implements error correction for the memories, caches, registers etc. Is <c>false</c> if the <see cref="DeviceOCL"/> does not implement error correction. This can be a requirement for certain clients of OpenCL. </value>
        public bool ErrorCorrectionSupport { get { return errorCorrectionSupport; } }

        /// <summary>
        /// Gets the <see cref="DeviceExecutionCapabilitiesOCL"/> of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The <see cref="DeviceExecutionCapabilitiesOCL"/> of the <see cref="DeviceOCL"/>. </value>
        public DeviceExecutionCapabilitiesOCL ExecutionCapabilities { get { return executionCapabilities; } }

        /// <summary>
        /// Gets a read-only collection of names of extensions that the <see cref="DeviceOCL"/> supports.
        /// </summary>
        /// <value> A read-only collection of names of extensions that the <see cref="DeviceOCL"/> supports. </value>
        public ReadOnlyCollection<string> Extensions { get { return extensions; } }

        /// <summary>
        /// Gets the size of the global <see cref="DeviceOCL"/> memory cache line in bytes.
        /// </summary>
        /// <value> The size of the global <see cref="DeviceOCL"/> memory cache line in bytes. </value>
        public long GlobalMemoryCacheLineSize { get { return globalMemoryCachelineSize; } }

        /// <summary>
        /// Gets the size of the global <see cref="DeviceOCL"/> memory cache in bytes.
        /// </summary>
        /// <value> The size of the global <see cref="DeviceOCL"/> memory cache in bytes. </value>
        public long GlobalMemoryCacheSize { get { return globalMemoryCacheSize; } }

        /// <summary>
        /// Gets the <see cref="DeviceMemoryCacheTypeOCL"/> of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The <see cref="DeviceMemoryCacheTypeOCL"/> of the <see cref="DeviceOCL"/>. </value>
        public DeviceMemoryCacheTypeOCL GlobalMemoryCacheType { get { return globalMemoryCacheType; } }

        /// <summary>
        /// Gets the size of the global <see cref="DeviceOCL"/> memory in bytes.
        /// </summary>
        /// <value> The size of the global <see cref="DeviceOCL"/> memory in bytes. </value>
        public long GlobalMemorySize { get { return globalMemorySize; } }

        /// <summary>
        /// Gets the maximum <see cref="Image2D.Height"/> value that the <see cref="DeviceOCL"/> supports in pixels.
        /// </summary>
        /// <value> The minimum value is 8192 if <see cref="DeviceOCL.ImageSupport"/> is <c>true</c>. </value>
        public long Image2DMaxHeight { get { return image2DMaxHeight; } }

        /// <summary>
        /// Gets the maximum <see cref="Image2D.Width"/> value that the <see cref="DeviceOCL"/> supports in pixels.
        /// </summary>
        /// <value> The minimum value is 8192 if <see cref="DeviceOCL.ImageSupport"/> is <c>true</c>. </value>
        public long Image2DMaxWidth { get { return image2DMaxWidth; } }

        /// <summary>
        /// Gets the maximum <see cref="Image3DOCL.Depth"/> value that the <see cref="DeviceOCL"/> supports in pixels.
        /// </summary>
        /// <value> The minimum value is 2048 if <see cref="DeviceOCL.ImageSupport"/> is <c>true</c>. </value>
        public long Image3DMaxDepth { get { return image3DMaxDepth; } }

        /// <summary>
        /// Gets the maximum <see cref="Image3DOCL.Height"/> value that the <see cref="DeviceOCL"/> supports in pixels.
        /// </summary>
        /// <value> The minimum value is 2048 if <see cref="DeviceOCL.ImageSupport"/> is <c>true</c>. </value>
        public long Image3DMaxHeight { get { return image3DMaxHeight; } }

        /// <summary>
        /// Gets the maximum <see cref="Image3DOCL.Width"/> value that the <see cref="DeviceOCL"/> supports in pixels.
        /// </summary>
        /// <value> The minimum value is 2048 if <see cref="DeviceOCL.ImageSupport"/> is <c>true</c>. </value>
        public long Image3DMaxWidth { get { return image3DMaxWidth; } }

        /// <summary>
        /// Gets the state of image support of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> Is <c>true</c> if <see cref="ImageOCL"/>s are supported by the <see cref="DeviceOCL"/> and <c>false</c> otherwise. </value>
        public bool ImageSupport { get { return imageSupport; } }

        /// <summary>
        /// Gets the size of local memory are of the <see cref="DeviceOCL"/> in bytes.
        /// </summary>
        /// <value> The minimum value is 16 KB (OpenCL 1.0) or 32 KB (OpenCL 1.1). </value>
        public long LocalMemorySize { get { return localMemorySize; } }

        /// <summary>
        /// Gets the <see cref="DeviceLocalMemoryTypeOCL"/> that is supported on the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The <see cref="DeviceLocalMemoryTypeOCL"/> that is supported on the <see cref="DeviceOCL"/>. </value>
        public DeviceLocalMemoryTypeOCL LocalMemoryType { get { return localMemoryType; } }

        /// <summary>
        /// Gets the maximum configured clock frequency of the <see cref="DeviceOCL"/> in MHz.
        /// </summary>
        /// <value> The maximum configured clock frequency of the <see cref="DeviceOCL"/> in MHz. </value>
        public long MaxClockFrequency { get { return maxClockFrequency; } }

        /// <summary>
        /// Gets the number of parallel compute cores on the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The minimum value is 1. </value>
        public long MaxComputeUnits { get { return maxComputeUnits; } }

        /// <summary>
        /// Gets the maximum number of arguments declared with the <c>__constant</c> or <c>constant</c> qualifier in a <see cref="KernelOCL"/> executing in the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The minimum value is 8. </value>
        public long MaxConstantArguments { get { return maxConstantArguments; } }

        /// <summary>
        /// Gets the maximum size in bytes of a constant buffer allocation in the <see cref="DeviceOCL"/> memory.
        /// </summary>
        /// <value> The minimum value is 64 KB. </value>
        public long MaxConstantBufferSize { get { return maxConstantBufferSize; } }

        /// <summary>
        /// Gets the maximum size of memory object allocation in the <see cref="DeviceOCL"/> memory in bytes.
        /// </summary>
        /// <value> The minimum value is <c>max(<see cref="DeviceOCL.GlobalMemorySize"/>/4, 128*1024*1024)</c>. </value>
        public long MaxMemoryAllocationSize { get { return maxMemAllocSize; } }

        /// <summary>
        /// Gets the maximum size in bytes of the arguments that can be passed to a <see cref="KernelOCL"/> executing in the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The minimum value is 256 (OpenCL 1.0) or 1024 (OpenCL 1.1). </value>
        public long MaxParameterSize { get { return maxParameterSize; } }

        /// <summary>
        /// Gets the maximum number of simultaneous <see cref="ImageOCL"/>s that can be read by a <see cref="KernelOCL"/> executing in the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The minimum value is 128 if <see cref="DeviceOCL.ImageSupport"/> is <c>true</c>. </value>
        public long MaxReadImageArguments { get { return maxReadImageArgs; } }

        /// <summary>
        /// Gets the maximum number of <see cref="SamplerOCL"/>s that can be used in a <see cref="KernelOCL"/>.
        /// </summary>
        /// <value> The minimum value is 16 if <see cref="DeviceOCL.ImageSupport"/> is <c>true</c>. </value>
        public long MaxSamplers { get { return maxSamplers; } }

        /// <summary>
        /// Gets the maximum number of work-items in a work-group executing a <see cref="KernelOCL"/> in a <see cref="DeviceOCL"/> using the data parallel execution model.
        /// </summary>
        /// <value> The minimum value is 1. </value>
        public long MaxWorkGroupSize { get { return maxWorkGroupSize; } }

        /// <summary>
        /// Gets the maximum number of dimensions that specify the global and local work-item IDs used by the data parallel execution model.
        /// </summary>
        /// <value> The minimum value is 3. </value>
        public long MaxWorkItemDimensions { get { return maxWorkItemDimensions; } }

        /// <summary>
        /// Gets the maximum number of work-items that can be specified in each dimension of the <paramref name="globalWorkSize"/> argument of <see cref="CommandQueueOCL.Execute"/>.
        /// </summary>
        /// <value> The maximum number of work-items that can be specified in each dimension of the <paramref name="globalWorkSize"/> argument of <see cref="CommandQueueOCL.Execute"/>. </value>
        public ReadOnlyCollection<long> MaxWorkItemSizes { get { return maxWorkItemSizes; } }

        /// <summary>
        /// Gets the maximum number of simultaneous <see cref="ImageOCL"/>s that can be written to by a <see cref="KernelOCL"/> executing in the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The minimum value is 8 if <see cref="DeviceOCL.ImageSupport"/> is <c>true</c>. </value>
        public long MaxWriteImageArguments { get { return maxWriteImageArgs; } }

        /// <summary>
        /// Gets the alignment in bits of the base address of any <see cref="MemoryOCL"/> allocated in the <see cref="DeviceOCL"/> memory.
        /// </summary>
        /// <value> The alignment in bits of the base address of any <see cref="MemoryOCL"/> allocated in the <see cref="DeviceOCL"/> memory. </value>
        public long MemoryBaseAddressAlignment { get { return memBaseAddrAlign; } }

        /// <summary>
        /// Gets the smallest alignment in bytes which can be used for any data type allocated in the <see cref="DeviceOCL"/> memory.
        /// </summary>
        /// <value> The smallest alignment in bytes which can be used for any data type allocated in the <see cref="DeviceOCL"/> memory. </value>
        public long MinDataTypeAlignmentSize { get { return minDataTypeAlignSize; } }

        /// <summary>
        /// Gets the name of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The name of the <see cref="DeviceOCL"/>. </value>
        public string Name { get { return name; } }

        /// <summary>
        /// Gets the <see cref="PlatformOCL"/> associated with the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The <see cref="PlatformOCL"/> associated with the <see cref="DeviceOCL"/>. </value>
        public PlatformOCL Platform { get { return platform; } }

        /// <summary>
        /// Gets the <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>char</c>s.
        /// </summary>
        /// <value> The <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>char</c>s. </value>
        /// <remarks> The vector width is defined as the number of scalar elements that can be stored in the vector. </remarks>
        public long PreferredVectorWidthChar { get { return preferredVectorWidthChar; } }

        /// <summary>
        /// Gets the <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>double</c>s or 0 if the <c>cl_khr_fp64</c> format is not supported.
        /// </summary>
        /// <value> The <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>double</c>s or 0 if the <c>cl_khr_fp64</c> format is not supported. </value>
        /// <remarks> The vector width is defined as the number of scalar elements that can be stored in the vector. </remarks>
        public long PreferredVectorWidthDouble { get { return GetInfo<uint>(DeviceInfoOCL.PreferredVectorWidthDouble); } }

        /// <summary>
        /// Gets the <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>float</c>s.
        /// </summary>
        /// <value> The <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>float</c>s. </value>
        /// <remarks> The vector width is defined as the number of scalar elements that can be stored in the vector. </remarks>
        public long PreferredVectorWidthFloat { get { return preferredVectorWidthFloat; } }

        /// <summary>
        /// Gets the <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>half</c>s or 0 if the <c>cl_khr_fp16</c> format is not supported.
        /// </summary>
        /// <value> The <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>half</c>s or 0 if the <c>cl_khr_fp16</c> format is not supported. </value>
        /// <remarks> The vector width is defined as the number of scalar elements that can be stored in the vector. </remarks>
        public long PreferredVectorWidthHalf { get { return GetInfo<uint>(DeviceInfoOCL.PreferredVectorWidthHalf); } }

        /// <summary>
        /// Gets the <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>int</c>s.
        /// </summary>
        /// <value> The <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>int</c>s. </value>
        /// <remarks> The vector width is defined as the number of scalar elements that can be stored in the vector. </remarks>
        public long PreferredVectorWidthInt { get { return preferredVectorWidthInt; } }

        /// <summary>
        /// Gets the <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>long</c>s.
        /// </summary>
        /// <value> The <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>long</c>s. </value>
        /// <remarks> The vector width is defined as the number of scalar elements that can be stored in the vector. </remarks>
        public long PreferredVectorWidthLong { get { return preferredVectorWidthLong; } }

        /// <summary>
        /// Gets the <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>short</c>s.
        /// </summary>
        /// <value> The <see cref="DeviceOCL"/>'s preferred native vector width size for vector of <c>short</c>s. </value>
        /// <remarks> The vector width is defined as the number of scalar elements that can be stored in the vector. </remarks>
        public long PreferredVectorWidthShort { get { return preferredVectorWidthShort; } }

        /// <summary>
        /// Gets the OpenCL profile name supported by the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> 
        /// The profile name returned can be one of the following strings:
        /// <list type="bullets">
        /// <item>
        ///     <term> FULL_PROFILE </term>
        ///     <description> The <see cref="DeviceOCL"/> supports the OpenCL specification (functionality defined as part of the core specification and does not require any extensions to be supported). </description>
        /// </item>
        /// <item>
        ///     <term> EMBEDDED_PROFILE </term>
        ///     <description> The <see cref="DeviceOCL"/> supports the OpenCL embedded profile. </description>
        /// </item>
        /// </list>
        /// </value>
        public string Profile { get { return profile; } }

        /// <summary>
        /// Gets the resolution of the <see cref="DeviceOCL"/> timer in nanoseconds.
        /// </summary>
        /// <value> The resolution of the <see cref="DeviceOCL"/> timer in nanoseconds. </value>
        public long ProfilingTimerResolution { get { return profilingTimerResolution; } }

        /// <summary>
        /// Gets the <see cref="DeviceSingleCapabilitiesOCL"/> of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The <see cref="DeviceSingleCapabilitiesOCL"/> of the <see cref="DeviceOCL"/>. </value>
        public DeviceSingleCapabilitiesOCL SingleCapabilites { get { return singleCapabilities; } }

        /// <summary>
        /// Gets the <see cref="DeviceTypesOCL"/> of the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The <see cref="DeviceTypesOCL"/> of the <see cref="DeviceOCL"/>. </value>
        public DeviceTypesOCL Type { get { return type; } }

        /// <summary>
        /// Gets the <see cref="DeviceOCL"/> vendor name string.
        /// </summary>
        /// <value> The <see cref="DeviceOCL"/> vendor name string. </value>
        public string Vendor { get { return vendor; } }

        /// <summary>
        /// Gets a unique <see cref="DeviceOCL"/> vendor identifier.
        /// </summary>
        /// <value> A unique <see cref="DeviceOCL"/> vendor identifier. </value>
        /// <remarks> An example of a unique device identifier could be the PCIe ID. </remarks>
        public long VendorId { get { return vendorId; } }

        /// <summary>
        /// Gets the OpenCL version supported by the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The OpenCL version supported by the <see cref="DeviceOCL"/>. </value>
        public Version Version { get { return ToolsOCL.ParseVersionString(VersionString, 1); } }

        /// <summary>
        /// Gets the OpenCL version string supported by the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The version string has the following format: <c>OpenCL[space][major_version].[minor_version][space][vendor-specific information]</c>. </value>
        public string VersionString { get { return version; } }

        //////////////////////////////////
        // OpenCL 1.1 device properties //
        //////////////////////////////////

        /// <summary>
        /// Gets information about the presence of the unified memory subsystem.
        /// </summary>
        /// <value> Is <c>true</c> if the <see cref="DeviceOCL"/> and the host have a unified memory subsystem and <c>false</c> otherwise. </value>
        /// <remarks> Requires OpenCL 1.1 </remarks>
        public bool HostUnifiedMemory { get { return GetBoolInfo(DeviceInfoOCL.HostUnifiedMemory); } }

        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>char</c>s.
        /// </summary>
        /// <value> The native ISA vector width size for vector of <c>char</c>s. </value>
        /// <remarks> 
        ///     <para> The vector width is defined as the number of scalar elements that can be stored in the vector. </para>
        ///     <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        public long NativeVectorWidthChar { get { return GetInfo<long>(DeviceInfoOCL.NativeVectorWidthChar); } }

        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>double</c>s or 0 if the <c>cl_khr_fp64</c> format is not supported.
        /// </summary>
        /// <value> The native ISA vector width size for vector of <c>double</c>s or 0 if the <c>cl_khr_fp64</c> format is not supported. </value>
        /// <remarks> 
        ///     <para> The vector width is defined as the number of scalar elements that can be stored in the vector. </para>
        ///     <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        public long NativeVectorWidthDouble { get { return GetInfo<long>(DeviceInfoOCL.NativeVectorWidthDouble); } }

        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>float</c>s.
        /// </summary>
        /// <value> The native ISA vector width size for vector of <c>float</c>s. </value>
        /// <remarks> 
        ///     <para> The vector width is defined as the number of scalar elements that can be stored in the vector. </para>
        ///     <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        public long NativeVectorWidthFloat { get { return GetInfo<long>(DeviceInfoOCL.NativeVectorWidthFloat); } }

        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>half</c>s or 0 if the <c>cl_khr_fp16</c> format is not supported.
        /// </summary>
        /// <value> The native ISA vector width size for vector of <c>half</c>s or 0 if the <c>cl_khr_fp16</c> format is not supported. </value>
        /// <remarks> 
        ///     <para> The vector width is defined as the number of scalar elements that can be stored in the vector. </para>
        ///     <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        public long NativeVectorWidthHalf { get { return GetInfo<long>(DeviceInfoOCL.NativeVectorWidthHalf); } }

        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>int</c>s.
        /// </summary>
        /// <value> The native ISA vector width size for vector of <c>int</c>s. </value>
        /// <remarks>
        ///     <para> The vector width is defined as the number of scalar elements that can be stored in the vector. </para>
        ///     <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        public long NativeVectorWidthInt { get { return GetInfo<long>(DeviceInfoOCL.NativeVectorWidthInt); } }

        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>long</c>s.
        /// </summary>
        /// <value> The native ISA vector width size for vector of <c>long</c>s. </value>
        /// <remarks>
        ///     <para> The vector width is defined as the number of scalar elements that can be stored in the vector. </para>
        ///     <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        public long NativeVectorWidthLong { get { return GetInfo<long>(DeviceInfoOCL.NativeVectorWidthLong); } }

        /// <summary>
        /// Gets the native ISA vector width size for vector of <c>short</c>s.
        /// </summary>
        /// <value> The native ISA vector width size for vector of <c>short</c>s. </value>
        /// <remarks> 
        ///     <para> The vector width is defined as the number of scalar elements that can be stored in the vector. </para>
        ///     <para> Requires OpenCL 1.1 </para>
        /// </remarks>
        public long NativeVectorWidthShort { get { return GetInfo<long>(DeviceInfoOCL.NativeVectorWidthShort); } }

        /// <summary>
        /// Gets the OpenCL C version supported by the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> Is <c>1.1</c> if <see cref="DeviceOCL.Version"/> is <c>1.1</c>. Is <c>1.0</c> or <c>1.1</c> if <see cref="DeviceOCL.Version"/> is <c>1.0</c>. </value>
        /// <remarks> Requires OpenCL 1.1. </remarks>
        public Version OpenCLCVersion { get { return ToolsOCL.ParseVersionString(OpenCLCVersionString, 2); } }

        /// <summary>
        /// Gets the OpenCL C version string supported by the <see cref="DeviceOCL"/>.
        /// </summary>
        /// <value> The OpenCL C version string supported by the <see cref="DeviceOCL"/>. The version string has the following format: <c>OpenCL[space]C[space][major_version].[minor_version][space][vendor-specific information]</c>. </value>
        /// <remarks> Requires OpenCL 1.1. </remarks>
        public string OpenCLCVersionString { get { return GetStringInfo(DeviceInfoOCL.OpenCLCVersion); } }

        #endregion

        #region Constructors

        internal DeviceOCL(PlatformOCL platform, CLDeviceHandle handle)
        {
            Handle = handle;
            SetID(Handle.Value);

            addressBits = GetInfo<uint>(DeviceInfoOCL.AddressBits);
            available = GetBoolInfo(DeviceInfoOCL.Available);
            compilerAvailable = GetBoolInfo(DeviceInfoOCL.CompilerAvailable);
            driverVersion = GetStringInfo(DeviceInfoOCL.DriverVersion);
            endianLittle = GetBoolInfo(DeviceInfoOCL.EndianLittle);
            errorCorrectionSupport = GetBoolInfo(DeviceInfoOCL.ErrorCorrectionSupport);
            executionCapabilities = (DeviceExecutionCapabilitiesOCL)GetInfo<long>(DeviceInfoOCL.ExecutionCapabilities);

            string extensionString = GetStringInfo(DeviceInfoOCL.Extensions);
            extensions = new ReadOnlyCollection<string>(extensionString.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));

            globalMemoryCachelineSize = GetInfo<uint>(DeviceInfoOCL.GlobalMemoryCachelineSize);
            globalMemoryCacheSize = (long)GetInfo<ulong>(DeviceInfoOCL.GlobalMemoryCacheSize);
            globalMemoryCacheType = (DeviceMemoryCacheTypeOCL)GetInfo<long>(DeviceInfoOCL.GlobalMemoryCacheType);
            globalMemorySize = (long)GetInfo<ulong>(DeviceInfoOCL.GlobalMemorySize);
            image2DMaxHeight = (long)GetInfo<IntPtr>(DeviceInfoOCL.Image2DMaxHeight);
            image2DMaxWidth = (long)GetInfo<IntPtr>(DeviceInfoOCL.Image2DMaxWidth);
            image3DMaxDepth = (long)GetInfo<IntPtr>(DeviceInfoOCL.Image3DMaxDepth);
            image3DMaxHeight = (long)GetInfo<IntPtr>(DeviceInfoOCL.Image3DMaxHeight);
            image3DMaxWidth = (long)GetInfo<IntPtr>(DeviceInfoOCL.Image3DMaxWidth);
            imageSupport = GetBoolInfo(DeviceInfoOCL.ImageSupport);
            localMemorySize = (long)GetInfo<ulong>(DeviceInfoOCL.LocalMemorySize);
            localMemoryType = (DeviceLocalMemoryTypeOCL)GetInfo<long>(DeviceInfoOCL.LocalMemoryType);
            maxClockFrequency = GetInfo<uint>(DeviceInfoOCL.MaxClockFrequency);
            maxComputeUnits = GetInfo<uint>(DeviceInfoOCL.MaxComputeUnits);
            maxConstantArguments = GetInfo<uint>(DeviceInfoOCL.MaxConstantArguments);
            maxConstantBufferSize = (long)GetInfo<ulong>(DeviceInfoOCL.MaxConstantBufferSize);
            maxMemAllocSize = (long)GetInfo<ulong>(DeviceInfoOCL.MaxMemoryAllocationSize);
            maxParameterSize = (long)GetInfo<IntPtr>(DeviceInfoOCL.MaxParameterSize);
            maxReadImageArgs = GetInfo<uint>(DeviceInfoOCL.MaxReadImageArguments);
            maxSamplers = GetInfo<uint>(DeviceInfoOCL.MaxSamplers);
            maxWorkGroupSize = (long)GetInfo<IntPtr>(DeviceInfoOCL.MaxWorkGroupSize);
            maxWorkItemDimensions = GetInfo<uint>(DeviceInfoOCL.MaxWorkItemDimensions);
            maxWorkItemSizes = new ReadOnlyCollection<long>(ToolsOCL.ConvertArray(GetArrayInfo<CLDeviceHandle, DeviceInfoOCL, IntPtr>(Handle, DeviceInfoOCL.MaxWorkItemSizes, CL10.GetDeviceInfo)));
            maxWriteImageArgs = GetInfo<uint>(DeviceInfoOCL.MaxWriteImageArguments);
            memBaseAddrAlign = GetInfo<uint>(DeviceInfoOCL.MemoryBaseAddressAlignment);
            minDataTypeAlignSize = GetInfo<uint>(DeviceInfoOCL.MinDataTypeAlignmentSize);
            name = GetStringInfo(DeviceInfoOCL.Name);
            this.platform = platform;
            preferredVectorWidthChar = GetInfo<uint>(DeviceInfoOCL.PreferredVectorWidthChar);
            preferredVectorWidthFloat = GetInfo<uint>(DeviceInfoOCL.PreferredVectorWidthFloat);
            preferredVectorWidthInt = GetInfo<uint>(DeviceInfoOCL.PreferredVectorWidthInt);
            preferredVectorWidthLong = GetInfo<uint>(DeviceInfoOCL.PreferredVectorWidthLong);
            preferredVectorWidthShort = GetInfo<uint>(DeviceInfoOCL.PreferredVectorWidthShort);
            profile = GetStringInfo(DeviceInfoOCL.Profile);
            profilingTimerResolution = (long)GetInfo<IntPtr>(DeviceInfoOCL.ProfilingTimerResolution);
            queueProperties = (CommandQueueFlagsOCL)GetInfo<long>(DeviceInfoOCL.CommandQueueProperties);
            singleCapabilities = (DeviceSingleCapabilitiesOCL)GetInfo<long>(DeviceInfoOCL.SingleFPConfig);
            type = (DeviceTypesOCL)GetInfo<long>(DeviceInfoOCL.Type);
            vendor = GetStringInfo(DeviceInfoOCL.Vendor);
            vendorId = GetInfo<uint>(DeviceInfoOCL.VendorId);
            version = GetStringInfo(DeviceInfoOCL.Version);
        }

        #endregion

        #region Private methods

        private bool GetBoolInfo(DeviceInfoOCL paramName)
        {
            return GetBoolInfo<CLDeviceHandle, DeviceInfoOCL>(Handle, paramName, CL10.GetDeviceInfo);
        }

        private NativeType GetInfo<NativeType>(DeviceInfoOCL paramName) where NativeType : struct
        {
            return GetInfo<CLDeviceHandle, DeviceInfoOCL, NativeType>(Handle, paramName, CL10.GetDeviceInfo);
        }

        private string GetStringInfo(DeviceInfoOCL paramName)
        {
            return GetStringInfo<CLDeviceHandle, DeviceInfoOCL>(Handle, paramName, CL10.GetDeviceInfo);
        }

        #endregion
    }
}
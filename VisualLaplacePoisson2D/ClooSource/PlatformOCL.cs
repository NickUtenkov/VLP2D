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
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using Cloo.Bindings;

    /// <summary>
    /// Represents an OpenCL platform.
    /// </summary>
    /// <remarks> The host plus a collection of devices managed by the OpenCL framework that allow an application to share resources and execute kernels on devices in the platform. </remarks>
    /// <seealso cref="DeviceOCL"/>
    /// <seealso cref="KernelOCL"/>
    /// <seealso cref="ResourceOCL"/>
    public class PlatformOCL : ObjectOCL
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ReadOnlyCollection<DeviceOCL> devices;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly ReadOnlyCollection<string> extensions;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string name;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private static ReadOnlyCollection<PlatformOCL> platforms;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string profile;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string vendor;

        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private readonly string version;

        #endregion

        #region Properties

        /// <summary>
        /// The handle of the <see cref="PlatformOCL"/>.
        /// </summary>
        public CLPlatformHandle Handle
        {
            get;
            protected set;
        }

        /// <summary>
        /// Gets a read-only collection of <see cref="DeviceOCL"/>s available on the <see cref="PlatformOCL"/>.
        /// </summary>
        /// <value> A read-only collection of <see cref="DeviceOCL"/>s available on the <see cref="PlatformOCL"/>. </value>
        public ReadOnlyCollection<DeviceOCL> Devices { get { return devices; } }

        /// <summary>
        /// Gets a read-only collection of extension names supported by the <see cref="PlatformOCL"/>.
        /// </summary>
        /// <value> A read-only collection of extension names supported by the <see cref="PlatformOCL"/>. </value>
        public ReadOnlyCollection<string> Extensions { get { return extensions; } }

        /// <summary>
        /// Gets the <see cref="PlatformOCL"/> name.
        /// </summary>
        /// <value> The <see cref="PlatformOCL"/> name. </value>
        public string Name { get { return name; } }

        /// <summary>
        /// Gets a read-only collection of available <see cref="PlatformOCL"/>s.
        /// </summary>
        /// <value> A read-only collection of available <see cref="PlatformOCL"/>s. </value>
        /// <remarks> The collection will contain no items, if no OpenCL platforms are found on the system. </remarks>
        public static ReadOnlyCollection<PlatformOCL> Platforms { get { return platforms; } }

        /// <summary>
        /// Gets the name of the profile supported by the <see cref="PlatformOCL"/>.
        /// </summary>
        /// <value> The name of the profile supported by the <see cref="PlatformOCL"/>. </value>
        public string Profile { get { return profile; } }

        /// <summary>
        /// Gets the <see cref="PlatformOCL"/> vendor.
        /// </summary>
        /// <value> The <see cref="PlatformOCL"/> vendor. </value>
        public string Vendor { get { return vendor; } }

        /// <summary>
        /// Gets the OpenCL version string supported by the <see cref="PlatformOCL"/>.
        /// </summary>
        /// <value> The OpenCL version string supported by the <see cref="PlatformOCL"/>. It has the following format: <c>OpenCL[space][major_version].[minor_version][space][vendor-specific information]</c>. </value>
        public string Version { get { return version; } }

        #endregion

        #region Constructors

        static PlatformOCL()
        {
            lock (typeof(PlatformOCL))
            {
                try
                {
                    if (platforms != null)
                        return;

                    CLPlatformHandle[] handles;
                    int handlesLength;
                    ErrorCodeOCL error = CL10.GetPlatformIDs(0, null, out handlesLength);
                    ExceptionOCL.ThrowOnError(error);
                    handles = new CLPlatformHandle[handlesLength];

                    error = CL10.GetPlatformIDs(handlesLength, handles, out handlesLength);
                    ExceptionOCL.ThrowOnError(error);

                    List<PlatformOCL> platformList = new List<PlatformOCL>(handlesLength);
                    foreach (CLPlatformHandle handle in handles)
                        platformList.Add(new PlatformOCL(handle));

                    platforms = platformList.AsReadOnly();
                }
                catch (DllNotFoundException)
                {
                    platforms = new List<PlatformOCL>().AsReadOnly();
                }
            }
        }

        private PlatformOCL(CLPlatformHandle handle)
        {
            Handle = handle;
            SetID(Handle.Value);

            string extensionString = GetStringInfo<CLPlatformHandle, PlatformInfoOCL>(Handle, PlatformInfoOCL.Extensions, CL10.GetPlatformInfo);
            extensions = new ReadOnlyCollection<string>(extensionString.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries));

            name = GetStringInfo<CLPlatformHandle, PlatformInfoOCL>(Handle, PlatformInfoOCL.Name, CL10.GetPlatformInfo);
            profile = GetStringInfo<CLPlatformHandle, PlatformInfoOCL>(Handle, PlatformInfoOCL.Profile, CL10.GetPlatformInfo);
            vendor = GetStringInfo<CLPlatformHandle, PlatformInfoOCL>(Handle, PlatformInfoOCL.Vendor, CL10.GetPlatformInfo);
            version = GetStringInfo<CLPlatformHandle, PlatformInfoOCL>(Handle, PlatformInfoOCL.Version, CL10.GetPlatformInfo);
            QueryDevices();
        }

        #endregion

        #region Public methods

        /// <summary>
        /// Gets a <see cref="PlatformOCL"/> of a specified handle.
        /// </summary>
        /// <param name="handle"> The handle of the queried <see cref="PlatformOCL"/>. </param>
        /// <returns> The <see cref="PlatformOCL"/> of the matching handle or <c>null</c> if none matches. </returns>
        public static PlatformOCL GetByHandle(IntPtr handle)
        {
            foreach (PlatformOCL platform in Platforms)
                if (platform.Handle.Value == handle)
                    return platform;

            return null;
        }

        /// <summary>
        /// Gets the first matching <see cref="PlatformOCL"/> of a specified name.
        /// </summary>
        /// <param name="platformName"> The name of the queried <see cref="PlatformOCL"/>. </param>
        /// <returns> The first <see cref="PlatformOCL"/> of the specified name or <c>null</c> if none matches. </returns>
        public static PlatformOCL GetByName(string platformName)
        {
            foreach (PlatformOCL platform in Platforms)
                if (platform.Name.Equals(platformName))
                    return platform;

            return null;
        }

        /// <summary>
        /// Gets the first matching <see cref="PlatformOCL"/> of a specified vendor.
        /// </summary>
        /// <param name="platformVendor"> The vendor of the queried <see cref="PlatformOCL"/>. </param>
        /// <returns> The first <see cref="PlatformOCL"/> of the specified vendor or <c>null</c> if none matches. </returns>
        public static PlatformOCL GetByVendor(string platformVendor)
        {
            foreach (PlatformOCL platform in Platforms)
                if (platform.Vendor.Equals(platformVendor))
                    return platform;

            return null;
        }

        /// <summary>
        /// Gets a read-only collection of available <see cref="DeviceOCL"/>s on the <see cref="PlatformOCL"/>.
        /// </summary>
        /// <returns> A read-only collection of the available <see cref="DeviceOCL"/>s on the <see cref="PlatformOCL"/>. </returns>
        /// <remarks> This method resets the <c>ComputePlatform.Devices</c>. This is useful if one or more of them become unavailable (<c>ComputeDevice.Available</c> is <c>false</c>) after a <see cref="ContextOCL"/> and <see cref="CommandQueueOCL"/>s that use the <see cref="DeviceOCL"/> have been created and commands have been queued to them. Further calls will trigger an <c>OutOfResourcesComputeException</c> until this method is executed. You will also need to recreate any <see cref="ResourceOCL"/> that was created on the no longer available <see cref="DeviceOCL"/>. </remarks>
        public ReadOnlyCollection<DeviceOCL> QueryDevices()
        {
            int handlesLength = 0;
            ErrorCodeOCL error = CL10.GetDeviceIDs(Handle, DeviceTypesOCL.All, 0, null, out handlesLength);
            ExceptionOCL.ThrowOnError(error);

            CLDeviceHandle[] handles = new CLDeviceHandle[handlesLength];
            error = CL10.GetDeviceIDs(Handle, DeviceTypesOCL.All, handlesLength, handles, out handlesLength);
            ExceptionOCL.ThrowOnError(error);

            DeviceOCL[] devices = new DeviceOCL[handlesLength];
            for (int i = 0; i < handlesLength; i++)
                devices[i] = new DeviceOCL(this, handles[i]);

            this.devices = new ReadOnlyCollection<DeviceOCL>(devices);

            return this.devices;
        }

        #endregion
    }
}
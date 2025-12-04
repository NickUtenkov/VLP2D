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
    using System.Runtime.InteropServices;

    public partial class CommandQueueOCL
    {
        #region CopyBuffer

        /// <summary>
        /// Enqueues a command to copy data from a source buffer to a destination buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffers. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBuffer<T>(BufferBaseOCL<T> source, BufferBaseOCL<T> destination, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, 0, 0, source.Count, events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source buffer to a destination buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffers. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBuffer<T>(BufferBaseOCL<T> source, BufferBaseOCL<T> destination, long sourceOffset, long destinationOffset, long region, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, sourceOffset, destinationOffset, region, events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source buffer to a destination buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffers. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBuffer<T>(BufferBaseOCL<T> source, BufferBaseOCL<T> destination, SysIntX2 sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, new SysIntX3(sourceOffset, 0), new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), 0, 0, 0, 0, events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source buffer to a destination buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffers. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBuffer<T>(BufferBaseOCL<T> source, BufferBaseOCL<T> destination, SysIntX3 sourceOffset, SysIntX3 destinationOffset, SysIntX3 region, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, sourceOffset, destinationOffset, region, 0, 0, 0, 0, events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source buffer to a destination buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffers. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="sourceRowPitch"> The size of a row of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationRowPitch"> The size of a row of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBuffer<T>(BufferBaseOCL<T> source, BufferBaseOCL<T> destination, SysIntX2 sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, long sourceRowPitch, long destinationRowPitch, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, new SysIntX3(sourceOffset, 0), new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), sourceRowPitch, 0, destinationRowPitch, 0, events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source buffer to a destination buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffers. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="sourceRowPitch"> The size of a row of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationRowPitch"> The size of a row of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="sourceSlicePitch"> The size of a 2D slice of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationSlicePitch"> The size of a 2D slice of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBuffer<T>(BufferBaseOCL<T> source, BufferBaseOCL<T> destination, SysIntX3 sourceOffset, SysIntX3 destinationOffset, SysIntX3 region, long sourceRowPitch, long destinationRowPitch, long sourceSlicePitch, long destinationSlicePitch, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, sourceOffset, destinationOffset, region, sourceRowPitch, sourceSlicePitch, destinationRowPitch, destinationSlicePitch, events);
        }

        #endregion

        #region CopyBufferToImage

        /// <summary>
        /// Enqueues a command to copy data from a buffer to an image.
        /// </summary>
        /// <typeparam name="T"> The type of data in <paramref name="source"/>. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The image to copy to. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBufferToImage<T>(BufferBaseOCL<T> source, ImageOCL destination, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, 0, new SysIntX3(), new SysIntX3(destination.Width, destination.Height, (destination.Depth == 0) ? 1 : destination.Depth), events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a buffer to an image.
        /// </summary>
        /// <typeparam name="T"> The type of data in <paramref name="source"/>. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The image to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBufferToImage<T>(BufferBaseOCL<T> source, Image2DOCL destination, long sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, sourceOffset, new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a buffer to an image.
        /// </summary>
        /// <typeparam name="T"> The type of data in <paramref name="source"/>. </typeparam>
        /// <param name="source"> The buffer to copy from. </param>
        /// <param name="destination"> The image to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyBufferToImage<T>(BufferBaseOCL<T> source, Image3DOCL destination, long sourceOffset, SysIntX3 destinationOffset, SysIntX3 region, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, sourceOffset, destinationOffset, region, events);
        }

        #endregion

        #region CopyImage

        /// <summary>
        /// Enqueues a command to copy data from a source image to a destination image.
        /// </summary>
        /// <param name="source"> The image to copy from. </param>
        /// <param name="destination"> The image to copy to. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyImage(ImageOCL source, ImageOCL destination, ICollection<EventBaseOCL> events)
        {
            Copy(source, destination, new SysIntX3(), new SysIntX3(), new SysIntX3(source.Width, source.Height, (source.Depth == 0) ? 1 : source.Depth), events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source image to a destination image.
        /// </summary>
        /// <param name="source"> The image to copy from. </param>
        /// <param name="destination"> The image to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyImage(Image2DOCL source, Image2DOCL destination, SysIntX2 sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, ICollection<EventBaseOCL> events)
        {
            Copy(source, destination, new SysIntX3(sourceOffset, 0), new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source image to a destination image.
        /// </summary>
        /// <param name="source"> The image to copy from. </param>
        /// <param name="destination"> The image to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyImage(Image2DOCL source, Image3DOCL destination, SysIntX2 sourceOffset, SysIntX3 destinationOffset, SysIntX2 region, ICollection<EventBaseOCL> events)
        {
            Copy(source, destination, new SysIntX3(sourceOffset, 0), destinationOffset, new SysIntX3(region, 1), events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source image to a destination image.
        /// </summary>
        /// <param name="source"> The image to copy from. </param>
        /// <param name="destination"> The image to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyImage(Image3DOCL source, Image2DOCL destination, SysIntX3 sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, ICollection<EventBaseOCL> events)
        {
            Copy(source, destination, sourceOffset, new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), null);
        }

        /// <summary>
        /// Enqueues a command to copy data from a source image to a destination image.
        /// </summary>
        /// <param name="source"> The image to copy from. </param>
        /// <param name="destination"> The image to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyImage(Image3DOCL source, Image3DOCL destination, SysIntX3 sourceOffset, SysIntX3 destinationOffset, SysIntX3 region, ICollection<EventBaseOCL> events)
        {
            Copy(source, destination, sourceOffset, destinationOffset, region, events);
        }

        #endregion

        #region CopyImageToBuffer

        /// <summary>
        /// Enqueues a command to copy data from an image to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in <paramref name="destination"/>. </typeparam>
        /// <param name="source"> The image to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyImageToBuffer<T>(ImageOCL source, BufferBaseOCL<T> destination, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, new SysIntX3(), 0, new SysIntX3(source.Width, source.Height, (source.Depth == 0) ? 1 : source.Depth), events);
        }

        /// <summary>
        /// Enqueues a command to copy data from an image to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in <paramref name="destination"/>. </typeparam>
        /// <param name="source"> The image to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyImageToBuffer<T>(Image2DOCL source, BufferBaseOCL<T> destination, SysIntX2 sourceOffset, long destinationOffset, SysIntX2 region, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, new SysIntX3(sourceOffset, 0), destinationOffset, new SysIntX3(region, 1), events);
        }

        /// <summary>
        /// Enqueues a command to copy data from a 3D image to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in <paramref name="destination"/>. </typeparam>
        /// <param name="source"> The image to copy from. </param>
        /// <param name="destination"> The buffer to copy to. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to copy. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void CopyImageToBuffer<T>(Image3DOCL source, BufferBaseOCL<T> destination, SysIntX3 sourceOffset, long destinationOffset, SysIntX3 region, ICollection<EventBaseOCL> events) where T : struct
        {
            Copy(source, destination, sourceOffset, destinationOffset, region, events);
        }

        #endregion

        #region ReadFromBuffer

        /// <summary>
        /// Enqueues a command to read data from a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The buffer to read from. </param>
        /// <param name="destination"> The array to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromBuffer<T>(BufferBaseOCL<T> source, ref T[] destination, bool blocking, IList<EventBaseOCL> events) where T : struct
        {
            ReadFromBuffer(source, ref destination, blocking, 0, 0, source.Count, events);
        }

        /// <summary>
        /// Enqueues a command to read data from a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The buffer to read from. </param>
        /// <param name="destination"> The array to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromBuffer<T>(BufferBaseOCL<T> source, ref T[] destination, bool blocking, long sourceOffset, long destinationOffset, long region, IList<EventBaseOCL> events) where T : struct
        {
            GCHandle destinationGCHandle = GCHandle.Alloc(destination, GCHandleType.Pinned);
            IntPtr destinationOffsetPtr = Marshal.UnsafeAddrOfPinnedArrayElement(destination, (int)destinationOffset);
            
            if (blocking)
            {
                Read(source, blocking, sourceOffset, region, destinationOffsetPtr, events);
                destinationGCHandle.Free();
            }
            else
            {
                bool userEventsWritable = (events != null && !events.IsReadOnly);
                IList<EventBaseOCL> eventList = (userEventsWritable) ? events : Events;
                Read(source, blocking, sourceOffset, region, destinationOffsetPtr, eventList);
                EventOCL newEvent = (EventOCL)eventList[eventList.Count - 1];
                newEvent.TrackGCHandle(destinationGCHandle);
            }
        }

        /// <summary>
        /// Enqueues a command to read data from a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The buffer to read from. </param>
        /// <param name="destination"> The array to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromBuffer<T>(BufferBaseOCL<T> source, ref T[,] destination, bool blocking, SysIntX2 sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, IList<EventBaseOCL> events) where T : struct
        {
            ReadFromBuffer(source, ref destination, blocking, sourceOffset, destinationOffset, region, 0, 0, events);
        }

        /// <summary>
        /// Enqueues a command to read data from a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The buffer to read from. </param>
        /// <param name="destination"> The array to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromBuffer<T>(BufferBaseOCL<T> source, ref T[, ,] destination, bool blocking, SysIntX3 sourceOffset, SysIntX3 destinationOffset, SysIntX3 region, IList<EventBaseOCL> events) where T : struct
        {
            ReadFromBuffer(source, ref destination, blocking, sourceOffset, destinationOffset, region, 0, 0, 0, 0, events);
        }

        /// <summary>
        /// Enqueues a command to read data from a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The buffer to read from. </param>
        /// <param name="destination"> The array to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="sourceRowPitch"> The size of a row of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationRowPitch"> The size of a row of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromBuffer<T>(BufferBaseOCL<T> source, ref T[,] destination, bool blocking, SysIntX2 sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, long sourceRowPitch, long destinationRowPitch, IList<EventBaseOCL> events) where T : struct
        {
            GCHandle destinationGCHandle = GCHandle.Alloc(destination, GCHandleType.Pinned);

            if (blocking)
            {
                Read(source, blocking, new SysIntX3(sourceOffset, 0), new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), sourceRowPitch, 0, destinationRowPitch, 0, destinationGCHandle.AddrOfPinnedObject(), events);
                destinationGCHandle.Free();
            }
            else
            {
                bool userEventsWritable = (events != null && !events.IsReadOnly);
                IList<EventBaseOCL> eventList = (userEventsWritable) ? events : Events;
                Read(source, blocking, new SysIntX3(sourceOffset, 0), new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), sourceRowPitch, 0, destinationRowPitch, 0, destinationGCHandle.AddrOfPinnedObject(), eventList);
                EventOCL newEvent = (EventOCL)eventList[eventList.Count - 1];
                newEvent.TrackGCHandle(destinationGCHandle);
            }
        }

        /// <summary>
        /// Enqueues a command to read data from a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The buffer to read from. </param>
        /// <param name="destination"> The array to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="sourceRowPitch"> The size of a row of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationRowPitch"> The size of a row of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="sourceSlicePitch"> The size of a 2D slice of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationSlicePitch"> The size of a 2D slice of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromBuffer<T>(BufferBaseOCL<T> source, ref T[, ,] destination, bool blocking, SysIntX3 sourceOffset, SysIntX3 destinationOffset, SysIntX3 region, long sourceRowPitch, long destinationRowPitch, long sourceSlicePitch, long destinationSlicePitch, IList<EventBaseOCL> events) where T : struct
        {
            GCHandle destinationGCHandle = GCHandle.Alloc(destination, GCHandleType.Pinned);

            if (blocking)
            {
                Read(source, blocking, sourceOffset, destinationOffset, region, sourceRowPitch, sourceSlicePitch, destinationRowPitch, destinationSlicePitch, destinationGCHandle.AddrOfPinnedObject(), events);
                destinationGCHandle.Free();
            }
            else
            {
                bool userEventsWritable = (events != null && !events.IsReadOnly);
                IList<EventBaseOCL> eventList = (userEventsWritable) ? events : Events;
                Read(source, blocking, sourceOffset, destinationOffset, region, sourceRowPitch, sourceSlicePitch, destinationRowPitch, destinationSlicePitch, destinationGCHandle.AddrOfPinnedObject(), eventList);
                EventOCL newEvent = (EventOCL)eventList[eventList.Count - 1];
                newEvent.TrackGCHandle(destinationGCHandle);
            }
        }

        #endregion

        #region ReadFromImage

        /// <summary>
        /// Enqueues a command to read data from an image.
        /// </summary>
        /// <param name="source"> The image to read from. </param>
        /// <param name="destination"> A valid pointer to a preallocated memory area to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromImage(ImageOCL source, IntPtr destination, bool blocking, ICollection<EventBaseOCL> events)
        {
            Read(source, blocking, new SysIntX3(), new SysIntX3(source.Width, source.Height, (source.Depth == 0) ? 1 : source.Depth), 0, 0, destination, events);
        }

        /// <summary>
        /// Enqueues a command to read data from an image.
        /// </summary>
        /// <param name="source"> The image to read from. </param>
        /// <param name="destination"> A valid pointer to a preallocated memory area to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromImage(Image2DOCL source, IntPtr destination, bool blocking, SysIntX2 sourceOffset, SysIntX2 region, ICollection<EventBaseOCL> events)
        {
            Read(source, blocking, new SysIntX3(sourceOffset, 0), new SysIntX3(region, 1), 0, 0, destination, events);
        }

        /// <summary>
        /// Enqueues a command to read data from an image.
        /// </summary>
        /// <param name="source"> The image to read from. </param>
        /// <param name="destination"> A valid pointer to a preallocated memory area to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromImage(Image3DOCL source, IntPtr destination, bool blocking, SysIntX3 sourceOffset, SysIntX3 region, ICollection<EventBaseOCL> events)
        {
            Read(source, blocking, sourceOffset, region, 0, 0, destination, events);
        }

        /// <summary>
        /// Enqueues a command to read data from an image.
        /// </summary>
        /// <param name="source"> The image to read from. </param>
        /// <param name="destination"> A valid pointer to a preallocated memory area to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="sourceRowPitch"> The size of a row of pixels of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromImage(Image2DOCL source, IntPtr destination, bool blocking, SysIntX2 sourceOffset, SysIntX2 region, long sourceRowPitch, ICollection<EventBaseOCL> events)
        {
            Read(source, blocking, new SysIntX3(sourceOffset, 0), new SysIntX3(region, 1), sourceRowPitch, 0, destination, events);
        }

        /// <summary>
        /// Enqueues a command to read data from an image.
        /// </summary>
        /// <param name="source"> The image to read from. </param>
        /// <param name="destination"> A valid pointer to a preallocated memory area to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="region"> The region of elements to read. </param>
        /// <param name="sourceRowPitch"> The size of a row of pixels of <paramref name="destination"/> in bytes. </param>
        /// <param name="sourceSlicePitch"> The size of a 2D slice of pixels of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void ReadFromImage(Image3DOCL source, IntPtr destination, bool blocking, SysIntX3 sourceOffset, SysIntX3 region, long sourceRowPitch, long sourceSlicePitch, ICollection<EventBaseOCL> events)
        {
            Read(source, blocking, sourceOffset, region, sourceRowPitch, sourceSlicePitch, destination, events);
        }

        #endregion

        #region WriteToBuffer

        /// <summary>
        /// Enqueues a command to write data to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The array to read from. </param>
        /// <param name="destination"> The buffer to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToBuffer<T>(T[] source, BufferBaseOCL<T> destination, bool blocking, IList<EventBaseOCL> events) where T : struct
        {
            WriteToBuffer(source, destination, blocking, 0, 0, destination.Count, events);
        }

        /// <summary>
        /// Enqueues a command to write data to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The array to read from. </param>
        /// <param name="destination"> The buffer to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToBuffer<T>(T[] source, BufferBaseOCL<T> destination, bool blocking, long sourceOffset, long destinationOffset, long region, IList<EventBaseOCL> events) where T : struct
        {
            GCHandle sourceGCHandle = GCHandle.Alloc(source, GCHandleType.Pinned);
            IntPtr sourceOffsetPtr = Marshal.UnsafeAddrOfPinnedArrayElement(source, (int)sourceOffset);

            if (blocking)
            {
                Write(destination, blocking, destinationOffset, region, sourceOffsetPtr, events);
                sourceGCHandle.Free();
            }
            else
            {
                bool userEventsWritable = (events != null && !events.IsReadOnly);
                IList<EventBaseOCL> eventList = (userEventsWritable) ? events : Events;
                Write(destination, blocking, destinationOffset, region, sourceOffsetPtr, eventList);
                EventOCL newEvent = (EventOCL)eventList[eventList.Count - 1];
                newEvent.TrackGCHandle(sourceGCHandle);
            }
        }

        /// <summary>
        /// Enqueues a command to write data to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The array to read from. </param>
        /// <param name="destination"> The buffer to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToBuffer<T>(T[,] source, BufferBaseOCL<T> destination, bool blocking, SysIntX2 sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, IList<EventBaseOCL> events) where T : struct
        {
            WriteToBuffer(source, destination, blocking, sourceOffset, destinationOffset, region, 0, 0, events);
        }

        /// <summary>
        /// Enqueues a command to write data to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The array to read from. </param>
        /// <param name="destination"> The buffer to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToBuffer<T>(T[, ,] source, BufferBaseOCL<T> destination, bool blocking, SysIntX3 sourceOffset, SysIntX3 destinationOffset, SysIntX3 region, IList<EventBaseOCL> events) where T : struct
        {
            WriteToBuffer(source, destination, blocking, sourceOffset, destinationOffset, region, 0, 0, 0, 0, events);
        }

        /// <summary>
        /// Enqueues a command to write data to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The array to read from. </param>
        /// <param name="destination"> The buffer to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="sourceRowPitch"> The size of a row of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationRowPitch"> The size of a row of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToBuffer<T>(T[,] source, BufferBaseOCL<T> destination, bool blocking, SysIntX2 sourceOffset, SysIntX2 destinationOffset, SysIntX2 region, long sourceRowPitch, long destinationRowPitch, IList<EventBaseOCL> events) where T : struct
        {
            GCHandle sourceGCHandle = GCHandle.Alloc(source, GCHandleType.Pinned);

            if (blocking)
            {
                Write(destination, blocking, new SysIntX3(sourceOffset, 0), new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), sourceRowPitch, 0, destinationRowPitch, 0, sourceGCHandle.AddrOfPinnedObject(), events);
                sourceGCHandle.Free();
            }
            else
            {
                bool userEventsWritable = (events != null && !events.IsReadOnly);
                IList<EventBaseOCL> eventList = (userEventsWritable) ? events : Events;
                Write(destination, blocking, new SysIntX3(sourceOffset, 0), new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), sourceRowPitch, 0, destinationRowPitch, 0, sourceGCHandle.AddrOfPinnedObject(), eventList);
                EventOCL newEvent = (EventOCL)eventList[eventList.Count - 1];
                newEvent.TrackGCHandle(sourceGCHandle);
            }
        }

        /// <summary>
        /// Enqueues a command to write data to a buffer.
        /// </summary>
        /// <typeparam name="T"> The type of data in the buffer. </typeparam>
        /// <param name="source"> The array to read from. </param>
        /// <param name="destination"> The buffer to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="sourceOffset"> The <paramref name="source"/> element position where reading starts. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="sourceRowPitch"> The size of a row of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationRowPitch"> The size of a row of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="sourceSlicePitch"> The size of a 2D slice of elements of <paramref name="source"/> in bytes. </param>
        /// <param name="destinationSlicePitch"> The size of a 2D slice of elements of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToBuffer<T>(T[, ,] source, BufferBaseOCL<T> destination, bool blocking, SysIntX3 sourceOffset, SysIntX3 destinationOffset, SysIntX3 region, long sourceRowPitch, long destinationRowPitch, long sourceSlicePitch, long destinationSlicePitch, IList<EventBaseOCL> events) where T : struct
        {
            GCHandle sourceGCHandle = GCHandle.Alloc(source, GCHandleType.Pinned);

            if (blocking)
            {
                Write(destination, blocking, sourceOffset, destinationOffset, region, sourceRowPitch, sourceSlicePitch, destinationRowPitch, destinationSlicePitch, sourceGCHandle.AddrOfPinnedObject(), events);
                sourceGCHandle.Free();
            }
            else
            {
                bool userEventsWritable = (events != null && !events.IsReadOnly);
                IList<EventBaseOCL> eventList = (userEventsWritable) ? events : Events;
                Write(destination, blocking, sourceOffset, destinationOffset, region, sourceRowPitch, sourceSlicePitch, destinationRowPitch, destinationSlicePitch, sourceGCHandle.AddrOfPinnedObject(), eventList);
                EventOCL newEvent = (EventOCL)eventList[eventList.Count - 1];
                newEvent.TrackGCHandle(sourceGCHandle);
            }
        }

        #endregion

        #region WriteToImage

        /// <summary>
        /// Enqueues a command to write data to an image.
        /// </summary>
        /// <param name="source"> A pointer to a memory area to read from. </param>
        /// <param name="destination"> The image to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToImage(IntPtr source, ImageOCL destination, bool blocking, ICollection<EventBaseOCL> events)
        {
            Write(destination, blocking, new SysIntX3(), new SysIntX3(destination.Width, destination.Height, (destination.Depth == 0) ? 1 : destination.Depth), 0, 0, source, events);
        }

        /// <summary>
        /// Enqueues a command to write data to an image.
        /// </summary>
        /// <param name="source"> A pointer to a memory area to read from. </param>
        /// <param name="destination"> The image to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToImage(IntPtr source, Image2DOCL destination, bool blocking, SysIntX2 destinationOffset, SysIntX2 region, ICollection<EventBaseOCL> events)
        {
            Write(destination, blocking, new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), 0, 0, source, events);
        }

        /// <summary>
        /// Enqueues a command to write data to an image.
        /// </summary>
        /// <param name="source"> A pointer to a memory area to read from. </param>
        /// <param name="destination"> The image to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToImage(IntPtr source, Image3DOCL destination, bool blocking, SysIntX3 destinationOffset, SysIntX3 region, ICollection<EventBaseOCL> events)
        {
            Write(destination, blocking, destinationOffset, region, 0, 0, source, events);
        }

        /// <summary>
        /// Enqueues a command to write data to an image.
        /// </summary>
        /// <param name="source"> A pointer to a memory area to read from. </param>
        /// <param name="destination"> The image to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="destinationRowPitch"> The size of a row of pixels of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToImage(IntPtr source, Image2DOCL destination, bool blocking, SysIntX2 destinationOffset, SysIntX2 region, long destinationRowPitch, ICollection<EventBaseOCL> events)
        {
            Write(destination, blocking, new SysIntX3(destinationOffset, 0), new SysIntX3(region, 1), destinationRowPitch, 0, source, events);
        }

        /// <summary>
        /// Enqueues a command to write data to an image.
        /// </summary>
        /// <param name="source"> A pointer to a memory area to read from. </param>
        /// <param name="destination"> The image to write to. </param>
        /// <param name="blocking"> The mode of operation of this command. If <c>true</c> this call will not return until the command has finished execution. </param>
        /// <param name="destinationOffset"> The <paramref name="destination"/> element position where writing starts. </param>
        /// <param name="region"> The region of elements to write. </param>
        /// <param name="destinationRowPitch"> The size of a row of pixels of <paramref name="destination"/> in bytes. </param>
        /// <param name="destinationSlicePitch"> The size of a 2D slice of pixels of <paramref name="destination"/> in bytes. </param>
        /// <param name="events"> A collection of events that need to complete before this particular command can be executed. If <paramref name="events"/> is not <c>null</c> a new event identifying this command is attached to the end of the collection. </param>
        public void WriteToImage(IntPtr source, Image3DOCL destination, bool blocking, SysIntX3 destinationOffset, SysIntX3 region, long destinationRowPitch, long destinationSlicePitch, ICollection<EventBaseOCL> events)
        {
            Write(destination, blocking, destinationOffset, region, destinationRowPitch, destinationSlicePitch, source, events);
        }

        #endregion
    }
}
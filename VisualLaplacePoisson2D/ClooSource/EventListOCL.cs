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
    using System.Collections;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using Cloo.Bindings;

    /// <summary>
    /// Represents a list of OpenCL generated or user created events.
    /// </summary>
    /// <seealso cref="CommandQueueOCL"/>
    public class EventListOCL : IList<EventBaseOCL>
    {
        #region Fields

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        private readonly List<EventBaseOCL> events;

        #endregion

        #region Constructors

        /// <summary>
        /// Creates an empty <see cref="EventListOCL"/>.
        /// </summary>
        public EventListOCL()
        {
            events = new List<EventBaseOCL>();
        }

        /// <summary>
        /// Creates a new <see cref="EventListOCL"/> from an existing list of <see cref="EventBaseOCL"/>s.
        /// </summary>
        /// <param name="events"> A list of <see cref="EventBaseOCL"/>s. </param>
        public EventListOCL(IList<EventBaseOCL> events)
        {
            events = new Collection<EventBaseOCL>(events);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the last <see cref="EventBaseOCL"/> on the list.
        /// </summary>
        /// <value> The last <see cref="EventBaseOCL"/> on the list. </value>
        public EventBaseOCL Last { get { return events[events.Count - 1]; } }

        #endregion

        #region Public methods

        /// <summary>
        /// Waits on the host thread for the specified events to complete.
        /// </summary>
        /// <param name="events"> The events to be waited for completition. </param>
        public static void Wait(ICollection<EventBaseOCL> events)
        {
            int eventWaitListSize;
            CLEventHandle[] eventHandles = ToolsOCL.ExtractHandles(events, out eventWaitListSize);
            ErrorCodeOCL error = CL10.WaitForEvents(eventWaitListSize, eventHandles);
            ExceptionOCL.ThrowOnError(error);
        }

        /// <summary>
        /// Waits on the host thread for the <see cref="EventBaseOCL"/>s in the <see cref="EventListOCL"/> to complete.
        /// </summary>
        public void Wait()
        {
            EventListOCL.Wait(events);
        }

        #endregion

        #region IList<ComputeEventBase> Members

        /// <summary>
        /// 
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public int IndexOf(EventBaseOCL item)
        {
            return events.IndexOf(item);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <param name="item"></param>
        public void Insert(int index, EventBaseOCL item)
        {
            events.Insert(index, item);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        public void RemoveAt(int index)
        {
            events.RemoveAt(index);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public EventBaseOCL this[int index]
        {
            get
            {
                return events[index];
            }
            set
            {
                events[index] = value;
            }
        }

        #endregion

        #region ICollection<ComputeEventBase> Members

        /// <summary>
        /// 
        /// </summary>
        /// <param name="item"></param>
        public void Add(EventBaseOCL item)
        {
            events.Add(item);
        }

        /// <summary>
        /// 
        /// </summary>
        public void Clear()
        {
            events.Clear();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Contains(EventBaseOCL item)
        {
            return events.Contains(item);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="array"></param>
        /// <param name="arrayIndex"></param>
        public void CopyTo(EventBaseOCL[] array, int arrayIndex)
        {
            events.CopyTo(array, arrayIndex);
        }

        /// <summary>
        /// 
        /// </summary>
        public int Count
        {
            get { return events.Count; }
        }

        /// <summary>
        /// 
        /// </summary>
        public bool IsReadOnly
        {
            get { return false; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="item"></param>
        /// <returns></returns>
        public bool Remove(EventBaseOCL item)
        {
            return events.Remove(item);
        }

        #endregion

        #region IEnumerable<ComputeEventBase> Members

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public IEnumerator<EventBaseOCL> GetEnumerator()
        {
            return ((IEnumerable<EventBaseOCL>)events).GetEnumerator();
        }

        #endregion

        #region IEnumerable Members

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)events).GetEnumerator();
        }

        #endregion
    }
}
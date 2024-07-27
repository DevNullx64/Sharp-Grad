using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// A memory management unit that manages memory buffers.
    /// </summary>
    public class MemoryManagementUnit : IBufferManager, ILowLevelMemoryManager
    {
        /// <summary>
        /// Tracks all allocated buffers.
        /// </summary>
        protected readonly List<AcceleratorBuffer> Allocs = [];

        /// <summary>
        /// The associated accelerator.
        /// </summary>
        public Accelerator Accelerator { get; }

        /// <summary>
        /// Instantiates a new <see cref="MemoryManagementUnit"/>.
        /// </summary>
        /// <param name="accelerator">The associated accelerator.</param>
        public MemoryManagementUnit(Accelerator accelerator)
        {
            Accelerator = accelerator;
        }

        /// <inheritdoc/>
        public AcceleratorBuffer<T> GetBuffer<T>(long length)
            where T : unmanaged
        {
            try
            {
                AcceleratorBuffer<T> buffer = new(this, length);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }
        
        /// <inheritdoc/>
        public AcceleratorBuffer<T> GetBuffer<T>(T[] values)
            where T : unmanaged
        {
            try
            {
                AcceleratorBuffer<T> buffer = new(this, values);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }
        
        /// <inheritdoc/>
        public AcceleratorBuffer<T> GetBuffer<T>(MemoryBuffer1D<T, Stride1D.Dense> data)
            where T : unmanaged
        {
            try
            {
                AcceleratorBuffer<T> buffer = new(this, data);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }

        /// <inheritdoc/>
        public void Release(AcceleratorBuffer buffer)
            => Allocs.Remove(buffer);

        /// <inheritdoc/>
        public long OffloadMemory(long length = 0)
        {
            lock (Allocs)
            {
                if (Allocs.Count == 0)
                    return 0;

                long toFree = length < 1 ? long.MaxValue : length;
                long Freed = 0;
                foreach (var buf in Allocs.Where(e => e.Location == BufferLocation.Accelerator).OrderBy(e => e.LastAccess))
                {
                    buf.Location = BufferLocation.Ram;
                    Freed += buf.Length;
                    if (Freed >= toFree)
                        break;
                }
                Synchronize();

                return Freed;
            }
        }

        /// <inheritdoc/>
        public void Synchronize() => Accelerator.Synchronize();

        // Kernel to fill a buffer with a value
        private static void FillKernel<T>(Index1D index1D, ArrayView<T> buffer, T value)
        where T : unmanaged
            { buffer[index1D] = value; }

        /// <inheritdoc/>
        public void Fill<T>(MemoryBuffer1D<T, Stride1D.Dense> acceleratorData, T value) where T : unmanaged
        {
            Action<Index1D, ArrayView<T>, T> kernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, T>(FillKernel);
            kernel((int)acceleratorData.Length, acceleratorData.View, value);
        }

        /// <inheritdoc/>
        MemoryBuffer1D<T, Stride1D.Dense> ILowLevelMemoryManager.MemoryBuffer1D<T>(long length)
        {
            int retry = 3;
            while (retry-- > 0)
            {
                try { return Accelerator.Allocate1D<T, Stride1D.Dense>(length, new Stride1D.Dense()); }
                catch { }
                OffloadMemory(length);
            }
            throw new OutOfMemoryException($"Failed to allocate memory for {length} elements.");
        }

        /// <inheritdoc/>
        MemoryBuffer1D<T, Stride1D.Dense> ILowLevelMemoryManager.MemoryBuffer1D<T>(T[] values)
        {
            var result = ((ILowLevelMemoryManager)this).MemoryBuffer1D<T>(values.LongLength);
            result.CopyFromCPU(values);
            return result;
        }
    }
}

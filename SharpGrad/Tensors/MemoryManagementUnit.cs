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
    /// Interface for low-level memory management.
    /// </summary>
    public interface ILowLevelMemoryManager
    {
        /// <summary>
        /// Allocates a 1D memory buffer.
        /// </summary>
        /// <typeparam name="T">The type of the elements.</typeparam>
        /// <param name="length">The length of the buffer.</param>
        /// <returns>The allocated memory buffer.</returns>
        /// <exception cref="OutOfMemoryException">Failed to allocate memory for the buffer.</exception>
        MemoryBuffer1D<T, Stride1D.Dense> MemoryBuffer1D<T>(long length)
            where T : unmanaged;

        /// <summary>
        /// Allocates a 1D memory buffer from the given values.
        /// </summary>
        /// <typeparam name="T">The type of the elements.</typeparam>
        /// <param name="values">The values to copy to the buffer.</param>
        /// <returns>The allocated memory buffer.</returns>
        /// <exception cref="OutOfMemoryException">Failed to allocate memory for the buffer.</exception>
        MemoryBuffer1D<T, Stride1D.Dense> MemoryBuffer1D<T>(T[] values)
            where T : unmanaged;

        /// <summary>
        /// Allocates a 1D memory buffer from the given data.
        /// </summary>
        /// <typeparam name="T">The type of the elements.</typeparam>
        /// <param name="data">The data to copy to the buffer.</param>
        /// <returns>The allocated memory buffer.</returns>
        /// <exception cref="OutOfMemoryException">Failed to allocate memory for the buffer.</exception>
        AcceleratorBuffer<T> GetBuffer<T>(MemoryBuffer1D<T, Stride1D.Dense> data)
            where T : unmanaged;

        /// <summary>
        /// Fills the given accelerator data with the specified value.
        /// </summary>
        /// <typeparam name="T">The type of the elements.</typeparam>
        /// <param name="acceleratorData">The accelerator data to fill.</param>
        /// <param name="value">The value to fill the data with.</param>
        void Fill<T>(MemoryBuffer1D<T, Stride1D.Dense> acceleratorData, T value)
            where T : unmanaged;
    }

    /// <summary>
    /// Interface for memory management.
    /// </summary>
    public interface IBufferManager
    {
        /// <summary>
        /// Gets an <see cref="AcceleratorBuffer"/> with the specified length.
        /// </summary>
        /// <typeparam name="T">The type of the elements.</typeparam>
        /// <param name="length">The length of the buffer.</param>
        /// <returns>The allocated buffer.</returns>
        /// <exception cref="OutOfMemoryException">Failed to allocate memory for the buffer.</exception>
        AcceleratorBuffer<T> GetBuffer<T>(long length)
            where T : unmanaged;

        /// <summary>
        /// Gets an <see cref="AcceleratorBuffer"/> from the given values.
        /// </summary>
        /// <typeparam name="T">The type of the elements.</typeparam>
        /// <param name="values">The values to copy to the buffer.</param>
        /// <returns>The allocated buffer.</returns>
        /// <exception cref="OutOfMemoryException">Failed to allocate memory for the buffer.</exception>
        AcceleratorBuffer<T> GetBuffer<T>(T[] values)
            where T : unmanaged;

        /// <summary>
        /// Releases the given buffer.
        /// </summary>
        void Release(AcceleratorBuffer buffer);

        /// <summary>
        /// Offloads memory from the accelerator to the RAM.
        /// </summary>
        /// <param name="length">The length of memory to offload.</param>
        /// <returns>The amount of memory offloaded.</returns>
        /// <remarks>Offloads the least recently used memory first.</remarks>
        long OffloadMemory(long length = 0);

        /// <summary>
        /// Synchronizes to pending operations.
        /// </summary>
        /// <remarks>Ensures that all operations are completed.</remarks>
        void Synchronize();
    }

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

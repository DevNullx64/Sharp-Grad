using ILGPU.Runtime;
using System;

namespace SharpGrad.Memory
{
    /// <summary>
    /// Interface for Buffers management.
    /// </summary>
    internal interface IBufferManager
    {
        /// <summary>
        /// Gets an <see cref="AcceleratorBuffer"/> with the specified length.
        /// </summary>
        /// <typeparam name="T">The type of the dataElements.</typeparam>
        /// <param name="length">The length of the buffer.</param>
        /// <returns>The allocated buffer.</returns>
        /// <exception cref="OutOfMemoryException">Failed to allocate Buffers for the buffer.</exception>
        AcceleratorBuffer<T> Allocate<T>(long length)
            where T : unmanaged;

        /// <summary>
        /// Gets an <see cref="AcceleratorBuffer"/> from the given values.
        /// </summary>
        /// <typeparam name="T">The type of the dataElements.</typeparam>
        /// <param name="values">The values to copy to the buffer.</param>
        /// <returns>The allocated buffer.</returns>
        /// <exception cref="OutOfMemoryException">Failed to allocate Buffers for the buffer.</exception>
        AcceleratorBuffer<T> Allocate<T>(T[] values)
            where T : unmanaged;

        /// <summary>
        /// Releases the given buffer.
        /// </summary>
        void Release(AcceleratorBuffer buffer);

        /// <summary>
        /// Offloads Buffers from the accelerator to the RAM.
        /// </summary>
        /// <param name="length">The length of Buffers to offload.</param>
        /// <returns>The amount of Buffers offloaded.</returns>
        /// <remarks>Offloads the least recently used Buffers first.</remarks>
        long OffloadMemory(long length = 0, Device? device = null);

        /// <summary>
        /// Synchronizes to pending operations.
        /// </summary>
        /// <remarks>Ensures that all operations are completed.</remarks>
        void Synchronize();
    }
}

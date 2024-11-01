using SharpGrad.Memory;
using System;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// Interface for memory management.
    /// </summary>
    internal interface IBufferManager
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
}

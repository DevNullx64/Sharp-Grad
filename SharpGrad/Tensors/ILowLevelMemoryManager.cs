using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;

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
}

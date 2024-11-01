using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Memory
{
    /// <summary>
    /// Internal interface for class that can create an <see cref="AcceleratorBuffer{T}"/>.
    /// </summary>
    /// <typeparam name="T">The type of the data.</typeparam>
    internal interface ICreateAcceleratorBuffer<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        /// <summary>
        /// Create a new <see cref="AcceleratorBuffer{T}"/> with the specified length.
        /// </summary>
        /// <param name="length">The length of the data.</param>
        /// <returns>A new <see cref="AcceleratorBuffer{T}"/> with the specified length.</returns>
        abstract static AcceleratorBuffer<T> Create(long length);

        /// <summary>
        /// Create a new <see cref="AcceleratorBuffer{T}"/> with the specified data.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <returns>A new <see cref="AcceleratorBuffer{T}"/> with the specified data.</returns>
        abstract static AcceleratorBuffer<T> Create(T[] data);

        /// <summary>
        /// Create a new <see cref="AcceleratorBuffer{T}"/> with the specified data.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <returns>A new <see cref="AcceleratorBuffer{T}"/> with the specified data.</returns>
        abstract static AcceleratorBuffer<T> Create(AcceleratorBuffer<T> data);

        /// <summary>
        /// Create a new <see cref="AcceleratorBuffer{T}"/> with the specified data.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <returns>A new <see cref="AcceleratorBuffer{T}"/> with the specified data.</returns>
        abstract static AcceleratorBuffer<T> Create(MemoryBuffer1D<T, Stride1D.Dense> data);
    }
}
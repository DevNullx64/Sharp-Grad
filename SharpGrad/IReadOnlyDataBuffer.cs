using System;

namespace SharpGrad
{
    /// <summary>
    /// Represents a read-only shaped array with shape, element type, and initialization status.
    /// </summary>
    public interface IReadOnlyDataBuffer
    {
        /// <summary>
        /// Gets the shape of the array.
        /// </summary>
        Shape Shape { get; }
        /// <summary>
        /// Gets the type of elements in the array.
        /// </summary>
        Type ElementType { get; }
        /// <summary>
        /// Gets a value indicating whether the array is initialized.
        /// </summary>
        bool IsInitialized { get; }
    }

    /// <summary>
    /// Represents a read-only shaped array of type T with indexing capabilities.
    /// </summary>
    /// <typeparam name="T">The element type of the array.</typeparam>
    public interface IReadOnlyDataBuffer<T> : IReadOnlyDataBuffer
    {
        /// <summary>
        /// Gets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        T this[params int[] indices] { get; }
        /// <summary>
        /// Gets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        T this[params Index[] indices] { get; }
        /// <summary>
        /// Gets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        T this[Dimdices indices] { get; }
    }
}
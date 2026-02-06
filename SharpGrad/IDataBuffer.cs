using System;

namespace SharpGrad
{
    /// <summary>
    /// Represents a mutable shaped array that can be initialized and freed.
    /// </summary>
    public interface IDataBuffer : IReadOnlyDataBuffer
    {
        /// <summary>
        /// Initializes the array if not already initialized.
        /// </summary>
        /// <returns>True if the array was initialized; otherwise, false.</returns>
        bool Initialize();

        /// <summary>
        /// Sets the data of the array.
        /// </summary>
        void SetData(Array newData);

        /// <summary>
        /// Frees the array resources.
        /// </summary>
        void Free();
    }

    /// <summary>
    /// Represents a mutable shaped array of type T with indexing and filling capabilities.
    /// </summary>
    /// <typeparam name="T">The element type of the array.</typeparam>
    public interface IDataBuffer<T> : IReadOnlyDataBuffer<T>
    {
        /// <summary>
        /// Gets or sets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        new T this[params int[] indices] { get; set; }

        /// <summary>
        /// Gets or sets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        new T this[params Index[] indices] { get; set; }

        /// <summary>
        /// Gets or sets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        new T this[Dimdices indices] { get; set; }

        /// <summary>
        /// Fills the array with the specified value.
        /// </summary>
        /// <param name="value">The value to fill the array with.</param>
        void Fill(T value);
    }

}
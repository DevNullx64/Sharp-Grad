using System;
using System.Collections.Generic;

namespace SharpGrad
{
    /// <summary>
    /// Interface for a shape of a tensor.
    /// </summary>
    public interface IShape : IReadOnlyList<int>
    {
        /// <summary>
        /// Gets the <see cref="int"/> at the specified index.
        /// </summary>
        /// <param name="index">The index of the <see cref="int"/> to get.</param>
        /// <returns>The <see cref="int"/> at the specified index.</returns>
        int this[Index index] { get; }

        /// <summary>
        /// Gets the <see cref="int"/>s at the specified range.
        /// </summary>
        /// <param name="range">The range of the <see cref="int"/>s to get.</param>
        /// <returns>The <see cref="int"/>s at the specified range.</returns>
        int[] this[Range range] { get; }

        /// <summary>
        /// Get the total number of dataElements in the shape.
        /// </summary>
        long Length { get; }

        /// <summary>
        /// Return true if the shape is a scalar.
        /// </summary>
        bool IsScalar { get; }

        /// <summary>
        /// Return the flattened index from the indices.
        /// </summary>
        /// <param name="indices">The indices to flatten.</param>
        /// <returns>The flattened index.</returns>
        int GetFlattenIndex(params Index[] indices);

        /// <summary>
        /// Return the indices from the flattened index.
        /// </summary>
        /// <param name="flattenedIndex">The flattened index to get the indices from.</param>
        /// <returns>The indices.</returns>
        Index[] GetIndices(int flattenedIndex);
    }
}

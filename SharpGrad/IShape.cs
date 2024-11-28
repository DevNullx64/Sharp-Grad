using System;
using System.Collections.Generic;

namespace SharpGrad
{
    /// <summary>
    /// Interface for a dimension of a tensor.
    /// </summary>
    public interface IShape : IReadOnlyList<Dimension>, IReadOnlySet<Dimension>, IEquatable<IShape>
    {
        /// <summary>
        /// Get ranges of <see cref="Dimension"/> from the dimension.
        /// </summary>
        /// <param name="ranges">The ranges of <see cref="Dimension"/> to get.</param>
        /// <returns>The ranges of the dimension.</returns>
        Shape this[params Range[] ranges] { get; }

        /// <summary>
        /// Get the total number of dataElements in the dimension.
        /// </summary>
        long Length { get; }

        /// <summary>
        /// Return true if the dimension is a scalar.
        /// </summary>
        bool IsScalar { get; }

        /// <summary>
        /// Return the flattened index from the indices.
        /// </summary>
        /// <param name="indices">The indices to flatten.</param>
        /// <returns>The flattened index.</returns>
        int GetFlattenIndex(params DimIndex[] indices);

        /// <summary>
        /// Return the indices from the flattened index.
        /// </summary>
        /// <param name="flattenedIndex">The flattened index to get the indices from.</param>
        /// <returns>The indices.</returns>
        DimIndex[] GetIndices(int flattenedIndex);
    }
}
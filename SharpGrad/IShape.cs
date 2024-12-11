using System;
using System.Collections.Generic;

namespace SharpGrad
{
    /// <summary>
    /// Interface for a dimension of a tensor.
    /// </summary>
    public interface IShape : IShapeBase, IReadOnlyList<Dimension>, IReadOnlySet<Dimension>
    {
        /// <summary>
        /// Get ranges of <see cref="Dimension"/> from the dimension.
        /// </summary>
        /// <param name="ranges">The ranges of <see cref="Dimension"/> to get.</param>
        /// <returns>The ranges of the dimension.</returns>
        Shape this[params Range[] ranges] { get; }

        /// <summary>
        /// Return the indices from the flattened index.
        /// </summary>
        /// <param name="flattenedIndex">The flattened index to get the indices from.</param>
        /// <returns>The indices.</returns>
        DimensionalIndex[] GetIndices(int flattenedIndex);
    }
}
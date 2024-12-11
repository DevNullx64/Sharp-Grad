using System;

namespace SharpGrad
{
    public interface IShapeBase : IEquatable<IShape>
    {
        /// <summary>
        /// Get the total number of dataElements in the dimension.
        /// </summary>
        long Rank { get; }

        /// <summary>
        /// Return true if the dimension is a scalar.
        /// </summary>
        bool IsScalar { get; }

        /// <summary>
        /// Return the flattened index from the indices.
        /// </summary>
        /// <param name="indices">The indices to flatten.</param>
        /// <returns>The flattened index.</returns>
        int GetOffset(params Index[] indices);
    }
}
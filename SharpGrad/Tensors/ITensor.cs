using System.Numerics;
using System;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// Abstract tensor interface.
    /// </summary>
    public interface ITensor : IEquatable<ITensor>
    {
        /// <summary>
        /// The shape of the tensor.
        /// </summary>
        Shape Shape { get; set; }

        /// <summary>
        /// The length of the tensor.
        /// </summary>
        long Length { get; }

        /// <summary>
        /// The computational depth of the tensor.
        /// </summary>
        long Depth { get; }
    }

    /// <summary>
    /// Interface for a tensor with generic type.
    /// </summary>
    public interface ITensor<T> : ITensor
    where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        /// <summary>
        /// Access the tensor element at the specified indices.
        /// </summary>
        T this[params Index[] indices] { get; }
    }
}

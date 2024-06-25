using System.Numerics;
using System;
using System.Collections.Generic;

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
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        /// <summary>
        /// The name of the tensor.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Access the tensor element at the specified indices.
        /// </summary>
        T this[params Index[] indices] { get; }

        /// <summary>
        /// Depth-first search to find the topological sort of the graph.
        /// </summary>
        /// <param name="topoSort">Topological sort of the graph.</param>
        /// <param name="visited">Set of visited tensors.</param>
        void DepthFirstSearch(Dictionary<Tensor<T>, int> topoSort);

    }
}

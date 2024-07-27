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

        /// <summary>
        /// The number of operands used by the tensor.
        /// </summary>
        int OperandCount { get; }
    }

    public class DfsNode<T>(Tensor<T> tensor, int index, int usageCount)
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        public Tensor<T> Tensor { get; } = tensor;
        public int Index { get; } = index;
        public int UsageCount { get; set; } = usageCount;
    }

    /// <summary>
    /// Interface for a tensor with generic type.
    /// </summary>
    public interface ITensor<T> : ITensor
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        /// <summary>
        /// The execution script of the tensor. Or forward only.
        /// </summary>
        /// <remarks>DOES NOT compute intermediate results. NOT usable with backpropagation.</remarks>
        KpuExecScript<T> ExecScript { get; }

        /// <summary>
        /// The forward script of the tensor.
        /// </summary>
        /// <remarks>Computes intermediate results. Usable with backpropagation.</remarks>
        KpuForwardScript<T> ForwardScript { get; }

        /// <summary>
        /// The backward script of the tensor.
        /// </summary>
        /// <remarks>Computes gradients.</remarks>
        KpuBackwardScript<T> BackwardScript { get; }
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
        /// <param name="needGradientOnly">Whether to search for tensors that need gradients only. Default is false.</param>
        Dictionary<Tensor<T>, DfsNode<T>> DepthFirstSearch(bool needGradientOnly = false);

    }
}

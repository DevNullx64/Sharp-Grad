using System;

namespace SharpGrad.ExprLambda
{
    /// <summary>
    /// Represents a node in a computation graph.
    /// </summary>
    public interface IGraphNode<T>
        where T : IGraphNode<T>
    {
        /// <summary>
        /// The data type of the node.
        /// </summary>
        Type Type { get; }

        /// <summary>
        /// The operands (child nodes) of this node.
        /// </summary>
        T[] Operands { get; }

        /// <summary>
        /// Indicates whether this node is the end of the graph.
        /// </summary>
        bool IsParallelBarrier { get; }
    }
}
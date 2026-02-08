using System;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Represents a typed node in a computation graph.
    /// Operands are accessible via OperandCount and GetOperand(int i).
    /// </summary>
    /// <typeparam name="N">The concrete node type that implements this interface.</typeparam>
    public interface IGraphNode<N> where N : IGraphNode<N>
    {
        /// <summary>
        /// The kind of this graph node.
        /// </summary>
        KindGraphNode Kind { get; }

        /// <summary>
        /// The element type of this graph node.
        /// </summary>
        Type ElementType { get; }

        /// <summary>
        /// The shape of this graph node.
        /// </summary>
        Shape Shape { get; }

        /// <summary>
        /// Indicates whether this graph node is an output node.
        /// </summary>
        bool IsOutput { get; set; }
    }
}
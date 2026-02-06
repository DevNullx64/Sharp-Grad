using SharpGrad.DifEngine.SyntaxBuilder.Operations;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Represents a binary operation node in a computation graph.
    /// </summary>
    /// <typeparam name="N">The concrete node type that implements this interface.</typeparam>
    public interface IGraphNodeBinary<N> : IGraphNode<N>
        where N : IGraphNode<N>
    {
        /// <summary>
        /// The kind of binary operation.
        /// </summary>
        new KindBinary Kind { get; }

        /// <summary>
        /// The left operand of the binary operation.
        /// </summary>
        N Left { get; }

        /// <summary>
        /// The right operand of the binary operation.
        /// </summary>
        N Right { get; }
    }
}
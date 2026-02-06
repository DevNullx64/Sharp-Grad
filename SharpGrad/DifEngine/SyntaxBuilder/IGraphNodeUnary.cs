using SharpGrad.DifEngine.SyntaxBuilder.Operations;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Represents a typed unary operation node in a computation graph.
    /// </summary>
    /// <typeparam name="N">The concrete node type that implements this interface.</typeparam>
    public interface IGraphNodeUnary<N> : IGraphNode<N>
        where N : IGraphNode<N>
    {
        /// <summary>
        /// Gets the kind of unary operation.
        /// </summary>
        new KindUnary Kind { get; }

        /// <summary>
        /// Gets the operand of the unary operation.
        /// </summary>
        N Operand { get; }
    }
}
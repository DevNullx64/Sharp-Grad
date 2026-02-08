namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Represents a typed reduction node in a computation graph.
    /// </summary>
    /// <typeparam name="N">The concrete node type that implements this interface.</typeparam>
    public interface IGraphNodeReduction<N> : IGraphNode<N>
        where N : IGraphNode<N>
    {
        /// <summary>
        /// Gets the kind of reduction operation.
        /// </summary>
        new KindReduction Kind { get; }

        /// <summary>
        /// Gets the operand of the reduction operation.
        /// </summary>
        N Operand { get; }

        /// <summary>
        /// Gets the dimensions over which the reduction is performed.
        /// </summary>
        Dimension[] Dimensions { get; }
    }
}
using SharpGrad.DifEngine.SyntaxBuilder.Operations;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    public interface IConditionalGraphNode<N> : IGraphNode<N>
        where N : IGraphNode<N>
    {
        new KindComparison Kind { get; }
        N Left { get; }
        N Right { get; }

        IFunctionGraphNode<N> TrueBranch { get; }
        IFunctionGraphNode<N> FalseBranch { get; }
    }
}
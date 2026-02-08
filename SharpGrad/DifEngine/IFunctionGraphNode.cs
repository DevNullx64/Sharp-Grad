using System;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    public interface IFunctionGraphNode<N> : IGraphNode<N>
    where N : IGraphNode<N>
    {
        new KindFunction Kind { get; }
        IGraph<N> Subgraph { get; }
    }
}
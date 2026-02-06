using SharpGrad.DifEngine.SyntaxBuilder.Operations;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    public interface IFunctionGraphNode<N> : IGraphNode<N>
    where N : IGraphNode<N>
    {
        new KindFunction Kind { get; }
        IGraph<N> Subgraph { get; }  // Le sous-graphe encapsulé
                                     // Autres propriétés si besoin (e.g., paramètres d'entrée)
    }
}
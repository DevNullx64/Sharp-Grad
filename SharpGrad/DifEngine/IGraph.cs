using System.Collections.Generic;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    /// <summary>
    /// Represents a computation graph consisting of nodes of type N.
    /// </summary>
    public interface IGraph<N> where N : IGraphNode<N>
    {
        /// <summary>
        /// Collection of all nodes in the graph.
        /// </summary>
        IEnumerable<N> Nodes { get; }

        /// <summary>
        /// Collection of leaf nodes (nodes without operands).
        /// </summary>
        IEnumerable<N> Leafs { get; }

        /// <summary>
        /// Collection of output nodes (marked IsOutput = true).
        /// </summary>
        IEnumerable<N> Outputs { get; }


        /// <summary>
        /// Gets the nodes in topological order.
        /// </summary>
        IEnumerable<N> GetTopologicalOrder();

        /// <summary>
        /// Checks if the graph contains the specified node.
        /// </summary>
        /// <param name="node">The node to check for.</param>
        /// <returns>True if the node is in the graph; otherwise, false.</returns>
        bool Contains(N node);
    }
}
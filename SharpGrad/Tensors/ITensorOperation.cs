using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorOperation : ITensor
    {
        OpCode OpCode { get; }
    }

    public interface ITensorOperation<T> : ITensorOperation, ITensor<T>
    where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {

        /// <summary>
        /// Depth-first search to find the topological sort of the graph.
        /// </summary>
        /// <param name="topoSort">Topological sort of the graph.</param>
        /// <param name="level">Computational level of the node.</param>
        /// <param name="visited">Dictionary of visited nodes. It contains the usage count and level of each node.</param>
        /// <param name="leaf">Dictionary of leaf nodes. It contains the location's count of each leaf node.</param>
        void DepthFirstSearch(List<ITensorOperation<T>> topoSort, int level, Dictionary<Tensor<T>, (int UsageCount, int Level)> visited, Dictionary<Tensor<T>, int> leaf);
        void Backward();
    }
}

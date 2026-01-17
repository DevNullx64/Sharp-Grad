using System;
using System.Collections.Generic;
using System.Linq.Expressions;

namespace SharpGrad.ExprLambda
{
    public static class ComputationGraphNodeExtensions
    {
        public static T[][] GetParallelSubgraphsDFS<T>(this T root) where T : IGraphNode<T>
        {
            Stack<T> reverseDFS = [];
            HashSet<T> visited = [];
            Stack<T> stack = new();
            List<T[]> allSubgraphs = [];

            // Iterative reverse DFS order traversal
            stack.Push(root);
            while (stack.Count > 0)
            {
                T current = stack.Pop();
                if (visited.Add(current))
                {
                    if (current.IsParallelBarrier && reverseDFS.Count != 0)
                    {
                        allSubgraphs.AddRange(GetParallelSubgraphsDFS(current));
                    }
                    else
                    {
                        foreach (var child in current.Operands)
                        {
                            stack.Push(child);
                        }
                    }
                    reverseDFS.Push(current);
                }
            }

            // Reverse to get correct order
            T[] thisDFS = new T[reverseDFS.Count];
            for (int i = 0; i < thisDFS.Length; i++)
            {
                thisDFS[i] = reverseDFS.Pop();
            }

            // Add the current DFS to the list of all subgraphs/DFS
            allSubgraphs.Add(thisDFS);

            return [.. allSubgraphs];
        }

        public static Expression BuildForwardExpression<T>(this T[][] graphs) where T : IGraphNode<T>
        {
            throw new NotImplementedException();
        }
        public static Expression BuildForwardExpression<T>(this T[] graphs) where T : IGraphNode<T>
        {
            if (graphs.Length == 0)
            {
                throw new ArgumentException("The graph array is empty.");
            }
            throw new NotImplementedException();
        }
    }
}

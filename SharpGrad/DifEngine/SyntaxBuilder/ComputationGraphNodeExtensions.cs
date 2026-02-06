using Microsoft.CodeAnalysis.CSharp;
using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    public static class ComputationGraphNodeExtensions
    {
        public static Value[][] GetParallelSubgraphsDFS(this Value root, Func<Value, bool> func)
        {
            Stack<Value> reverseDFS = [];
            HashSet<Value> visited = [];
            Stack<Value> stack = new();
            List<Value[]> allSubgraphs = [];

            // Iterative reverse DFS order traversal
            stack.Push(root);
            while (stack.Count > 0)
            {
                Value current = stack.Pop();
                if (visited.Add(current))
                {
                    if (func(current) && reverseDFS.Count != 0)
                    {
                        allSubgraphs.AddRange(GetParallelSubgraphsDFS(current, func));
                    }
                    else
                    {
                        if (current.Kind.IsUnary())
                        {
                            IGraphNodeUnary<Value> unaryNode = (IGraphNodeUnary<Value>)current;
                            stack.Push(unaryNode.Operand);
                        }
                        else if (current.Kind.IsBinary())
                        {
                            IGraphNodeBinary<Value> binaryNode = (IGraphNodeBinary<Value>)current;
                            stack.Push(binaryNode.Right);
                            stack.Push(binaryNode.Left);
                        }
                        else
                        {
                            throw new NotSupportedException($"Unsupported graph node kind: {current.Kind}");
                        }
                        reverseDFS.Push(current);
                    }
                }
            }

            // Reverse to get correct order
            Value[] thisDFS = new Value[reverseDFS.Count];
            for (int i = 0; i < thisDFS.Length; i++)
            {
                thisDFS[i] = reverseDFS.Pop();
            }

            // Add the current DFS to the list of all subgraphs/DFS
            allSubgraphs.Add(thisDFS);

            return [.. allSubgraphs];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Value[] GetDFS(this Value root)
            => root.GetParallelSubgraphsDFS(node => false)[0];
    }
}

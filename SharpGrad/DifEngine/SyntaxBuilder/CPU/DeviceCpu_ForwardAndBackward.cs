using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder.CPU
{
    public partial class DeviceCpu
    {
        /// <summary>
        /// Computes the forward pass from the given root <see cref="Value"/> node.
        /// </summary>
        /// <param name="root">The root Value node of the graph to compute.</param>
        /// <remarks>
        /// This method execute the forward pass of all subgraphs built from the specified root node.
        /// </remarks>
        public void Forward(Value root)
        {
            if (root.Kind.IsValue())
            {
                return;
            }

            Value[][] dfs = root.GetParallelSubgraphsDFS(n => n.Kind.IsReduction());
            for (int i = 0; i < dfs.Length; i++)
            {
                ForwardDFS(dfs[i]);
            }
        }

        /// <summary>
        /// Computes the forward pass for a given depth-first search (DFS) array of <see cref="Value"/> nodes.
        /// </summary>
        /// <param name="dfs">An array of Value nodes representing the DFS traversal of the computation graph.</param>
        /// <remarks>
        /// This method processes each node in the provided DFS array to compute the results
        /// based on the operations defined in the graph.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ForwardDFS(Value[] dfs)
        {
            for (int i = 0; i < dfs.Length; i++)
            {
                var node = dfs[i];
                if (!node.Kind.IsValue())
                {
                    switch (node)
                    {
                        case IGraphNodeUnary<Value> unaryNode:
                            Type outputElement = unaryNode.ElementType;
                            Type inputElement = unaryNode.Operand.ElementType;

                            // Check if node is an operation of mixed types or not
                            if (outputElement == inputElement)
                            {
                                ExecuteUnaryForward(unaryNode.Operand, node);
                            }
                            else
                            {
                                // Currently, only Cast operations support different input and output types
                                if (unaryNode.Kind != KindUnary.Cast)
                                {
                                    throw new InvalidOperationException($"{node.Kind} operation requires matching input and output types, but got {inputElement} and {outputElement}.");
                                }
                                ExecuteCastForward(unaryNode.Operand, node);
                            }
                            break;
                        case IGraphNodeBinary<Value> binaryNode:
                            ExecuteBinaryForward(binaryNode.Kind, binaryNode.Left, binaryNode.Right, node);
                            break;
                        case IGraphNodeReduction<Value> reductionNode:
                            throw new NotImplementedException($"Backward for GraphNodeKind {node.Kind} is not implemented.");
                        case IFunctionGraphNode<Value> functionNode:
                            throw new NotImplementedException($"Computation for GraphNodeKind {node.Kind} is not implemented.");
                        default:
                            throw new NotImplementedException($"Computation for GraphNodeKind {node.Kind} is not implemented.");
                    }
                }
            }
        }


        /// <summary>
        /// Computes the backward pass from the given root <see cref="Value"/> node.
        /// </summary>
        /// <param name="root">The root Value node of the graph to compute.</param>
        /// <remarks>
        /// This method execute the backward pass of all subgraphs built from the specified root node in reverse order.
        /// </remarks>
        public void Backward(Value root)
        {
            if (root.Kind.IsValue())
            {
                return;
            }
            Value[][] dfs = root.GetParallelSubgraphsDFS(n => n.Kind.IsReduction());
            for (int i = dfs.Length - 1; i >= 0; i--)
            {
                BackwardDFS(dfs[i]);
            }
        }

        /// <summary>
        /// Computes the backward pass for a given depth-first search (DFS) array of <see cref="Value"/> nodes.
        /// </summary>
        /// <param name="dfs">An array of Value nodes representing the DFS traversal of the computation graph.</param>
        /// <remarks>
        /// This method processes each node in the provided DFS array in reverse order to compute the gradients
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void BackwardDFS(Value[] dfs)
        {
            for (int i = dfs.Length - 1; i >= 0; i--)
            {
                var node = dfs[i];
                if (!node.Kind.IsValue())
                {
                    switch (node)
                    {
                        case IGraphNodeUnary<Value> unaryNode:
                            if (!unaryNode.Operand.IsGradiable)
                                continue;

                            Type outputElement = unaryNode.ElementType;
                            Type inputElement = unaryNode.Operand.ElementType;
                            // Check if node is an operation of mixed types or not
                            if (outputElement == inputElement)
                            {
                                ExecuteUnaryBackward(unaryNode.Operand, node);
                            }
                            else
                            {
                                // Currently, only Cast operations support different input and output types
                                if (unaryNode.Kind != KindUnary.Cast)
                                {
                                    throw new InvalidOperationException($"{node.Kind} operation requires matching input and output types, but got {inputElement} and {outputElement}.");
                                }
                                ExecuteCastBackward(unaryNode.Operand, node);
                            }
                            break;
                        case IGraphNodeBinary<Value> binaryNode:
                            if (!binaryNode.Left.IsGradiable && !binaryNode.Right.IsGradiable)
                                continue;
                            ExecuteBinaryBackward(binaryNode.Kind, binaryNode.Left, binaryNode.Right, node);
                            break;
                        case IGraphNodeReduction<Value> reductionNode:
                            throw new NotImplementedException($"Backward for GraphNodeKind {node.Kind} is not implemented.");
                        case IFunctionGraphNode<Value> functionNode:
                            throw new NotImplementedException($"Backward for GraphNodeKind {node.Kind} is not implemented.");
                        default:
                            throw new NotImplementedException($"Backward for GraphNodeKind {node.Kind} is not implemented.");
                    }
                }
            }
        }
    }
}

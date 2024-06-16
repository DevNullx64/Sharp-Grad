using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class StreamTensor1<T, TOp>(Tensor<T> operand1)
        : Tensor<T, TOp>(operand1.Shape), ITensorOperation1<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor1<T, T>
    {
        public Tensor<T> Operand1 => operand1;

        public override long Depth => operand1.Depth + 1;

        public override T this[params Index[] indices] => TOp.Exec(operand1[indices]);

        public override void DepthFirstSearch(List<ITensorOperation<T>> topoSort, int level, Dictionary<Tensor<T>, (int UsageCount, int Level)> visited, Dictionary<Tensor<T>, int> leaf)
        {
            if (visited.TryGetValue(this, out var count1))
            {
                visited[this] = (count1.UsageCount + 1, Math.Max(count1.Level, level));
            }
            else
            {
                visited.Add(this, (1, level));
                if (Operand1 is ITensorOperation<T> op)
                    op.DepthFirstSearch(topoSort, level + 1, visited, leaf);
                else
                {
                    if (leaf.TryGetValue(Operand1, out var count2))
                        leaf[Operand1] = count2 + 1;
                    else
                        leaf.Add(Operand1, 1);
                }
                topoSort.Add(this);
            }
        }

        public override void Backward()
        {
            if (Operand1 is ITensorOperation<T> op)
                op.Backward();
        }

        public override bool Equals(ITensor? other)
             => other is StreamTensor1<T, TOp> tensor && Operand1.Equals(tensor.Operand1);
    }
}

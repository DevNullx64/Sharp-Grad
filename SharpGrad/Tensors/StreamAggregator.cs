using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class StreamAggregator<T, TOp>(Tensor<T> operand1)
        : Tensor<T, TOp>(operand1.Shape), ITensorReduce<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IAggregator<T, TOp>
    {
        public Tensor<T> Operand1 => throw new NotImplementedException();

        public override long Depth => Operand1.Depth + 1;

        public override T this[params Index[] indices] => throw new NotImplementedException();

        public override void DepthFirstSearch(List<ITensorOperation<T>> topoSort, int level, Dictionary<Tensor<T>, (int UsageCount, int Level)> visited, Dictionary<Tensor<T>, int> leaf)
        {
            if (visited.TryGetValue(this, out var count1))
            {
                visited[this] = (count1.UsageCount + 1, Math.Max(count1.Level, level));
            }
            else
            {
                visited.Add(this, (1, level));
                if (Operand1 is ITensorOperation<T> op1)
                    op1.DepthFirstSearch(topoSort, level + 1, visited, leaf);
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
            if (Operand1 is ITensorOperation<T> op1)
                op1.Backward();
        }

        public override bool Equals(ITensor? other)
            => other is StreamAggregator<T, TOp> aggregator && Operand1.Equals(aggregator.Operand1);
    }
}

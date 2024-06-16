using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class StreamTensor2<T, TOp>(Tensor<T> operand1, Tensor<T> operand2)
        : Tensor<T, TOp>(operand1.Shape), ITensorOperation2<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor2<T, T, T>
    {
        public Tensor<T> Operand1 => operand1;
        public Tensor<T> Operand2 => operand2;

        public override long Depth { get; } = Math.Max(operand1.Depth, operand2.Depth) + 1;

        public override T this[params Index[] indices] => TOp.Exec(operand1[indices], operand2[indices]);

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
                if (Operand2 is ITensorOperation<T> op2)
                    op2.DepthFirstSearch(topoSort, level + 1, visited, leaf);
                else
                {
                    if (leaf.TryGetValue(Operand2, out var count3))
                        leaf[Operand2] = count3 + 1;
                    else
                        leaf.Add(Operand2, 1);
                }
                topoSort.Add(this);
            }
        }

        public override void Backward()
        {
            if (Operand1 is ITensorOperation<T> op1)
                op1.Backward();
            if (Operand2 is ITensorOperation<T> op2)
                op2.Backward();
        }

        public override bool Equals(ITensor? other) => other is StreamTensor2<T, TOp> tensor &&
            (
                Operand1.Equals(tensor.Operand1) && Operand2.Equals(tensor.Operand2) ||
                TOp.OpCode.HasFlag(OpCode.Commutative) && Operand1.Equals(tensor.Operand2) && Operand2.Equals(tensor.Operand1)
            );
    }
}

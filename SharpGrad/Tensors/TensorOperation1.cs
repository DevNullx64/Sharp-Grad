using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorOperation1<T, TOp>(Tensor<T> operand1)
        : TensorOperation<T, TOp>(operand1.Shape), ITensorOperation1<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IExecutor1<T, T>
    {
        public Tensor<T> Operand1 => operand1;

        public override long Depth { get; } = operand1.Depth + 1;

        public override int OperandCound => 1;

        public override T this[params Index[] indices] => TOp.Exec(operand1[indices]);

        internal override void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort)
        {
            if (topoSort.TryGetValue(this, out DfsNode<T>? count))
                count.UsageCount++;
            else
            {
                Operand1.DepthFirstSearch(topoSort);
                topoSort.Add(this, new(this, topoSort.Count, 1));
            }
        }

        public override void Backward()
        {
            if (Operand1 is ITensorOperation<T> op)
                op.Backward();
        }

        public override bool Equals(ITensor? other)
             => other is TensorOperation1<T, TOp> tensor && Operand1.Equals(tensor.Operand1);

        public override bool Equals(object? obj) => obj is TensorOperation1<T, TOp> tensor && Operand1.Equals(tensor.Operand1);
        public override int GetHashCode() => ((typeof(TOp).GetHashCode() * 31) + Operand1.GetHashCode()) * 31;

        public override string ToString() => $"{Name}({Operand1})";

        public static bool operator ==(TensorOperation1<T, TOp> left, TensorOperation1<T, TOp> right) => left.Equals(right);
        public static bool operator !=(TensorOperation1<T, TOp> left, TensorOperation1<T, TOp> right) => !left.Equals(right);
    }
}

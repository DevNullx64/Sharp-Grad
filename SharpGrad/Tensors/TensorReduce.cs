using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorReduce<T, TOp>(Tensor<T> operand1)
        : Tensor<T, TOp>(operand1.Shape), ITensorReduce<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IAggregator<T, TOp>
    {
        public Tensor<T> Operand1 { get; } = operand1;

        public override long Depth { get; } = operand1.Depth + 1;

        public override int OperandCound => -1;

        public override T this[params Index[] indices] => throw new NotImplementedException();

        internal override void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort)
        {
            if (!topoSort.TryGetValue(this, out DfsNode<T>? _))
            {
                Operand1.DepthFirstSearch(topoSort);
                topoSort.Add(this, new(this, topoSort.Count, 1));
            }
        }

        public override void Backward()
        {
            if (Operand1 is ITensorOperation<T> op1)
                op1.Backward();
        }

        public override bool Equals(ITensor? other)
            => other is TensorReduce<T, TOp> aggregator && Operand1.Equals(aggregator.Operand1);
    }
}

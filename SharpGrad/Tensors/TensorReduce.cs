using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorReduce<T, TOp>(Tensor<T> operand, Index? dim = null)
        : TensorOperation<T, TOp>(operand.Shape), ITensorReduce<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IAggregator<T, TOp>
    {
        public override Shape Shape { 
            get => Operand.Shape.Reduce(Dim);
            set => throw new NotSupportedException($"Impossible to set shape of {GetType().Name}");
        }

        public Tensor<T> Operand { get; } = operand;

        public Index Dim = dim ?? new Index(0, true);
        public override long Depth { get; } = operand.Depth + 1;

        public override int OperandCound => -1;

        public override T this[params Index[] indices] => throw new NotFiniteNumberException();

        internal override void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort)
        {
            if (!topoSort.TryGetValue(this, out DfsNode<T>? _))
            {
                Operand.DepthFirstSearch(topoSort);
                topoSort.Add(this, new(this, topoSort.Count, 1));
            }
        }

        public override void Backward()
        {
            if (Operand is ITensorOperation<T> op1)
                op1.Backward();
        }

        public override bool Equals(ITensor? other)
            => other is TensorReduce<T, TOp> aggregator && Operand.Equals(aggregator.Operand);
    }
}

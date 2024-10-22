using SharpGrad.Tensors.Operators;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorOperation1<T, TOp>
        : TensorOperation<T, TOp>, ITensorOperation1<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IExecUnary<T, T>
    {
        public Tensor<T> Operand { get; }

        public override long Depth { get; }

        public override bool NeedsGradient => Operand.NeedsGradient;

        public override int OperandCount => 1;

        public TensorOperation1(Tensor<T> operand)
            : base(TOp.ResultingShape(operand.Shape))
        {
            Operand = operand;
            Depth = operand.Depth + 1;
        }
        internal override void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort, DepthFirstSearchOption needGradientOnly = DepthFirstSearchOption.None)
        {
            if (topoSort.TryGetValue(this, out DfsNode<T>? count))
                count.UsageCount++;
            else if (needGradientOnly.HasFlag(DepthFirstSearchOption.AllGradient) || NeedsGradient)
            {
                Operand.DepthFirstSearch(topoSort, needGradientOnly);
                topoSort.Add(this, new(this, topoSort.Count, 1));
            }
        }

        public override void Backward()
        {
            if (Operand is ITensorOperation<T> op)
                op.Backward();
        }

        public override bool Equals(ITensor? other)
             => other is TensorOperation1<T, TOp> tensor && Operand.Equals(tensor.Operand);

        public override bool Equals(object? obj) => obj is TensorOperation1<T, TOp> tensor && Operand.Equals(tensor.Operand);
        public override int GetHashCode() => ((typeof(TOp).GetHashCode() * 31) + Operand.GetHashCode()) * 31;

        public override string ToString() => $"{Name}({Operand})";

        public static bool operator ==(TensorOperation1<T, TOp> left, TensorOperation1<T, TOp> right) => left.Equals(right);
        public static bool operator !=(TensorOperation1<T, TOp> left, TensorOperation1<T, TOp> right) => !left.Equals(right);
    }
}

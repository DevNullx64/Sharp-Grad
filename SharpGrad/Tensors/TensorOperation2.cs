using SharpGrad.Tensors.KPU;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorOperation2<T, TOp>(Tensor<T> operand1, Tensor<T> operand2)
        : TensorOperation<T, TOp>(TOp.ResultingShape(operand1.Shape, operand2.Shape)), ITensorOperation2<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IExecBinary<T, T, T>
    {
        public Tensor<T> Left { get; } = operand1;
        public Tensor<T> Right { get; } = operand2;

        public override long Depth { get; } = Math.Max(operand1.Depth, operand2.Depth) + 1;

        public override bool NeedsGradient => Left.NeedsGradient || Right.NeedsGradient;

        public override int OperandCount => 2;

        internal override void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort, DepthFirstSearchOption options = DepthFirstSearchOption.None)
        {
            if (topoSort.TryGetValue(this, out DfsNode<T>? node))
                node.UsageCount++;
            else if(options.HasFlag(DepthFirstSearchOption.AllGradient)
                 || NeedsGradient)
            {
                if (options.HasFlag(DepthFirstSearchOption.LowDepthFirst)
                 || Left.Depth >= Right.Depth)
                {
                    Left.DepthFirstSearch(topoSort, options);
                    Right.DepthFirstSearch(topoSort, options);
                }
                else
                {
                    Right.DepthFirstSearch(topoSort, options);
                    Left.DepthFirstSearch(topoSort, options);
                }
                topoSort.Add(this, new(this, topoSort.Count, 1));
            }
        }

        public override void Backward()
        {
            if (Left is ITensorOperation<T> op1)
                op1.Backward();
            if (Right is ITensorOperation<T> op2)
                op2.Backward();
        }

        public bool Equals(OpCode opCode, ITensor? left, ITensor? right)
            => OpCode == opCode &&
            (
                Left.Equals(left) && Right.Equals(right) ||
                (TOp.OpCode.HasFlag(OpCode.IsCommutative) && Left.Equals(right) && Right.Equals(left))
            );

        public override bool Equals(ITensor? other) => other is TensorOperation2<T, TOp> tensor &&
            (
                Left.Equals(tensor.Left) && Right.Equals(tensor.Right) ||
                TOp.OpCode.HasFlag(OpCode.IsCommutative) && Left.Equals(tensor.Right) && Right.Equals(tensor.Left)
            );

        public override bool Equals(object? obj) => obj is TensorOperation2<T, TOp> tensor && Equals(tensor);
        public override int GetHashCode(){
            if (TOp.OpCode.HasFlag(OpCode.IsCommutative))
                return ((typeof(TOp).GetHashCode() * 31 + Left.GetHashCode()) * 31 + Right.GetHashCode()) * 31;
            return typeof(TOp).GetHashCode() * 31 + (Left.GetHashCode() * 31 + Right.GetHashCode() * 31) * 31;
        }

        public override string ToString() => $"({Left} {Name} {Right})";

        public static bool operator ==(TensorOperation2<T, TOp> left, TensorOperation2<T, TOp> right) => left.Equals(right);
        public static bool operator !=(TensorOperation2<T, TOp> left, TensorOperation2<T, TOp> right) => !left.Equals(right);
    }
}

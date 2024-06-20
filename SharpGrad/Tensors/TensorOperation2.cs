using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorOperation2<T, TOp>(Tensor<T> operand1, Tensor<T> operand2)
        : Tensor<T, TOp>(operand1.Shape), ITensorOperation2<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
        where TOp : IExecutor2<T, T, T>
    {
        public Tensor<T> Operand1 => operand1;
        public Tensor<T> Operand2 => operand2;

        public override long Depth { get; } = Math.Max(operand1.Depth, operand2.Depth) + 1;

        public override T this[params Index[] indices] => TOp.Exec(operand1[indices], operand2[indices]);

        public override void DepthFirstSearch(List<Tensor<T>> topoSort, HashSet<Tensor<T>> visited)
        {
            if (visited.Add(this))
            {
                visited.Add(this);
                if (Operand1 is ITensorOperation<T> op1)
                    op1.DepthFirstSearch(topoSort, visited);
                if (Operand2 is ITensorOperation<T> op2)
                    op2.DepthFirstSearch(topoSort, visited);
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

        public override bool Equals(ITensor? other) => other is TensorOperation2<T, TOp> tensor &&
            (
                Operand1.Equals(tensor.Operand1) && Operand2.Equals(tensor.Operand2) ||
                TOp.OpCode.HasFlag(OpCode.Commutative) && Operand1.Equals(tensor.Operand2) && Operand2.Equals(tensor.Operand1)
            );

        public override bool Equals(object? obj) => obj is TensorOperation2<T, TOp> tensor && Equals(tensor);
        public override int GetHashCode(){
            if (TOp.OpCode.HasFlag(OpCode.Commutative))
                return ((typeof(TOp).GetHashCode() * 31 + Operand1.GetHashCode()) * 31 + Operand2.GetHashCode()) * 31;
            return typeof(TOp).GetHashCode() * 31 + (Operand1.GetHashCode() * 31 + Operand2.GetHashCode() * 31) * 31;
        }

        public override string ToString() => $"({Operand1} {TOp.OpCode} {Operand2})";

        public static bool operator ==(TensorOperation2<T, TOp> left, TensorOperation2<T, TOp> right) => left.Equals(right);
        public static bool operator !=(TensorOperation2<T, TOp> left, TensorOperation2<T, TOp> right) => !left.Equals(right);
    }
}

using SharpGrad.Memory;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorOperation2<T, TOp>(Tensor<T> operand1, Tensor<T> operand2)
        : TensorOperation<T, TOp>(operand1.Shape), ITensorOperation2<T, TOp>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IExecutor2<T, T, T>
    {
        public Tensor<T> Operand1 => operand1;
        public Tensor<T> Operand2 => operand2;

        public override long Depth { get; } = Math.Max(operand1.Depth, operand2.Depth) + 1;

        public override int OperandCound => 2;

        protected AcceleratorBuffer<T>? buffer = null;
        public override T this[params Index[] indices]{
            get
            {
                if (buffer is null)
                {
                    KernelProcessUnit kpu = KernelProcessUnit.DefaultKPU;
                    kpu.Exec(this);
                    buffer = kpu.GetBuffer(kpu.Exec(this).buffer);
                }
                return buffer[Shape.GetFlattenIndex(indices)];
            }
        }

        internal override void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort)
        {
            if (!topoSort.TryGetValue(this, out DfsNode<T>? _))
            {
                if (Operand1.Depth >= Operand2.Depth)
                {
                    Operand1.DepthFirstSearch(topoSort);
                    Operand2.DepthFirstSearch(topoSort);
                }
                else
                {
                    Operand2.DepthFirstSearch(topoSort);
                    Operand1.DepthFirstSearch(topoSort);
                }
                topoSort.Add(this, new(this, topoSort.Count, 1));
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

        public override string ToString() => $"({Operand1} {Name} {Operand2})";

        public static bool operator ==(TensorOperation2<T, TOp> left, TensorOperation2<T, TOp> right) => left.Equals(right);
        public static bool operator !=(TensorOperation2<T, TOp> left, TensorOperation2<T, TOp> right) => !left.Equals(right);
    }
}

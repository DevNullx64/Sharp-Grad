using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorOpGrad1<T, TOp>(Tensor<T> operand1) : TensorGrad<T>(TOp.ResultingShape(operand1.Shape)), ITensorOpBackward<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IOperation11_1<T>
    {
        public readonly Tensor<T> Operand1 = operand1;

        public override T this[params Index[] indices]
        {
            get => data[Shape.GetFlattenIndex(indices)];
            set => throw new NotSupportedException($"Cannot set value of {GetType().Name}");
        }

        public void Backward()
        {
            if(Operand1 is ITensorOpBackward<T> op1)
                op1.Backward();
        }

        public void DepthFirstSearch(HashSet<Tensor<T>> visited, Stack<Tensor<T>> stack)
        {
            if(visited.Add(this))
            {
                if(Operand1 is ITensorOpBackward<T> op1)
                    op1.DepthFirstSearch(visited, stack);
                stack.Push(this);
            }
        }
    }
}

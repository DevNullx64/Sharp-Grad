using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class TensorOp2<T, TOp>(Tensor<T> operand1, Tensor<T> operand2) : Tensor<T>(TOp.ResultingShape(operand1.Shape, operand2.Shape))
        where T : unmanaged, INumber<T>
        where TOp : IOperation11_2<T>
    {
        public readonly Tensor<T> Operand1 = operand1;
        public readonly Tensor<T> Operand2 = operand2;

        private AcceleratorBuffer<T>? data = null;
        internal AcceleratorBuffer<T> Data {
            get
            {
                if (data is null)
                {
                    data = Acc.GetAcceleratorBuffer<T>(Shape.Length);
                    Acc.Exec(TOp.Exec, Operand1.GetArrayView1D(), Operand2.GetArrayView1D(), GetArrayView1D());
                }
                return data;
            }
        }
        public override T this[params Index[] indices]
        {
            get => Data[Shape.GetFlattenIndex(indices)];
            set => throw new NotSupportedException($"Cannot set value of {GetType().Name}");
        }

        internal override ArrayView1D<T, Stride1D.Dense> GetArrayView1D() => Data.AcceleratorData.View;
    }
}

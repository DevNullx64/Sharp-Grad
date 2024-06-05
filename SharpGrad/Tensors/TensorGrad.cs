using System;
using System.Numerics;
using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;

namespace SharpGrad.Tensors
{
    public abstract class TensorGrad<T>(Shape shape) : TensorData<T>(shape), ITensorGrad<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        public readonly AcceleratorBuffer<T> Gradients = Acc.GetAcceleratorBuffer<T>(shape.Length);

        public void AddGrad(AcceleratorBuffer<T> grad)
            => Acc.Exec(AddOp<T>.Exec, Gradients.AcceleratorData, grad.AcceleratorData, Gradients.AcceleratorData);
        public void ApplyGrad(T lr)
        {
            Acc.Exec(MulOp<T>.Exec, Gradients.AcceleratorData, lr, Gradients.AcceleratorData);
            Acc.Exec(SubOp<T>.Exec, data.AcceleratorData, Gradients.AcceleratorData, data.AcceleratorData);
        }
    }
}

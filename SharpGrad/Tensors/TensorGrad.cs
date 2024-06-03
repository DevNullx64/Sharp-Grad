using System.Numerics;

namespace SharpGrad.Tensors
{
    public abstract class TensorGrad<T>(Shape shape) : TensorData<T>(shape), ITensorGrad<T>
        where T : unmanaged, IFloatingPoint<T>
    {
        protected readonly AcceleratorBuffer<T> grad = new(shape.Length);

        public void AddGrad(AcceleratorBuffer<T> grad)
            => Acc.Exec(AddOp<T>.Exec, this.grad.AcceleratorData, grad.AcceleratorData, this.grad.AcceleratorData);
        public void ApplyGrad(T lr)
        {
            Acc.Exec(MulOp<T>.Exec, grad.AcceleratorData, lr, grad.AcceleratorData);
            Acc.Exec(SubOp<T>.Exec, data.AcceleratorData, grad.AcceleratorData, data.AcceleratorData);
        }
    }
}

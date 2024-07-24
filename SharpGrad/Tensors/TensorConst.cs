using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class TensorConst<T> : Tensor<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        public override long Depth => 0;
        public override int OperandCount => 0;

        public override bool NeedsGradient => false;

        protected TensorConst(string name, Shape shape, AcceleratorBuffer<T> buffer)
            : base(name, shape, buffer)
        { }
        public TensorConst(string name, Shape shape, T[] data)
            : this(name, shape, KernelProcessUnit.DefaultKPU.MMU.GetBuffer(data))
        { }

        public override bool Equals(ITensor? other)
            => other is TensorConst<T> tensor && Buffer == tensor.Buffer;

        public override bool Equals(object? obj) => obj is TensorData<T> tensor && Equals(tensor);

        public override string ToString() => $"{Name}{Shape}";

        public override void Backward()
        {
            throw new NotSupportedException($"TensorData<{typeof(T).Name}> does not support backward operation");
        }

    }
}

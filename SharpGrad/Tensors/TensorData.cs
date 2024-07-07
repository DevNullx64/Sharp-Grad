using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class TensorData<T> : Tensor<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        internal readonly AcceleratorBuffer<T> buffer;
        internal ArrayView1D<T, Stride1D.Dense> View => buffer.AcceleratorData.View;

        public override long Depth => 0;

        public override int OperandCount => 0;

        public override T this[params Index[] indices]
        {
            get
            {
                var flattenedIndex = Shape.GetFlattenIndex(indices);
                return buffer[flattenedIndex];
            }
        }
        public void Set(T value, params Index[] indices)
        {
            var flattenedIndex = Shape.GetFlattenIndex(indices);
            buffer[flattenedIndex] = value;
        }

        internal TensorData(string name, Shape shape, AcceleratorBuffer<T> buffer)
            : base(name, shape)
        {
            Shape = shape;
            this.buffer = buffer;
        }
        protected TensorData(Shape shape, AcceleratorBuffer<T> buffer)
            :this(GetNextName(), shape, buffer) { }


        public TensorData(string name, Shape shape)
            : this(name, shape, KernelProcessUnit.DefaultKPU.MMU.GetBuffer<T>(shape.Length)) { }
        public TensorData(Shape shape)
            : this(GetNextName(), shape) { }

        public TensorData(string name, Shape shape, T[] data)
            : this(name, shape, KernelProcessUnit.DefaultKPU.MMU.GetBuffer(data)) { }
        public TensorData(Shape shape, T[] data)
            : this(GetNextName(), shape, data) { }

        public override bool Equals(ITensor? other)
            => other is TensorData<T> tensor && buffer == tensor.buffer;

        public override bool Equals(object? obj) => obj is TensorData<T> tensor && Equals(tensor);

        public override string ToString() => $"{Name}{Shape}";

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public static implicit operator TensorData<T>((string Name, Shape Shape) tensor) => new(tensor.Name, tensor.Shape);
        public static implicit operator TensorData<T>((string Name, Shape Shape, T[] Data) tensor) => new(tensor.Name, tensor.Shape, tensor.Data);

        public static implicit operator TensorData<T>((Shape Shape, string Name) tensor) => new(tensor.Name, tensor.Shape);
        public static implicit operator TensorData<T>((Shape Shape, string Name, T[] Data) tensor) => new(tensor.Name, tensor.Shape, tensor.Data);
    }
}

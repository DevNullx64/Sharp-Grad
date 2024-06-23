using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class DataTensor<T> : Tensor<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        protected readonly AcceleratorBuffer<T> buffer;
        internal ArrayView1D<T, Stride1D.Dense> View => buffer.AcceleratorData.View;

        public override long Depth => 0;

        public override T this[params Index[] indices]
        {
            get
            {
                var flattenedIndex = Shape.FlattenFrom(Shape, indices);
                return buffer[flattenedIndex];
            }
        }
        public void Set(T value, params Index[] indices)
        {
            var flattenedIndex = Shape.FlattenFrom(Shape, indices);
            buffer[flattenedIndex] = value;
        }

        protected DataTensor(string name, Shape shape, AcceleratorBuffer<T> buffer)
            : base(name, shape)
        {
            Shape = shape;
            this.buffer = buffer;
        }
        protected DataTensor(Shape shape, AcceleratorBuffer<T> buffer)
            :this(GetNextName(), shape, buffer) { }


        public DataTensor(string name, Shape shape)
            : this(name, shape, KernelProcessUnit.DefaultKPU.GetBuffer<T>(shape.Length)) { }
        public DataTensor(Shape shape)
            : this(GetNextName(), shape) { }

        public DataTensor(string name, Shape shape, T[] data)
            : this(name, shape, KernelProcessUnit.DefaultKPU.GetBuffer(data)) { }
        public DataTensor(Shape shape, T[] data)
            : this(GetNextName(), shape, data) { }

        public override bool Equals(ITensor? other)
            => other is DataTensor<T> tensor && buffer == tensor.buffer;

        public override bool Equals(object? obj) => obj is DataTensor<T> tensor && Equals(tensor);

        public override void Backward()
        {
            throw new NotImplementedException();
        }

        public static implicit operator DataTensor<T>((string Name, Shape Shape) tensor) => new(tensor.Name, tensor.Shape);
        public static implicit operator DataTensor<T>((string Name, Shape Shape, T[] Data) tensor) => new(tensor.Name, tensor.Shape, tensor.Data);

        public static implicit operator DataTensor<T>((Shape Shape, string Name) tensor) => new(tensor.Name, tensor.Shape);
        public static implicit operator DataTensor<T>((Shape Shape, string Name, T[] Data) tensor) => new(tensor.Name, tensor.Shape, tensor.Data);
    }
}

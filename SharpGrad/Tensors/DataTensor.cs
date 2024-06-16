using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class DataTensor<T> : Tensor<T>
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

        public DataTensor(Shape shape) : base(shape)
        {
            Shape = shape;
            buffer = KernelProcessUnit.DefaultKPU.GetBuffer<T>(shape.Length);
        }

        public DataTensor(Shape shape, T[] data) : base(shape)
        {
            if (shape.Length != data.Length)
                throw new InvalidOperationException($"Invalid data length {data.Length} for shape {shape}");
            Shape = shape;
            buffer = KernelProcessUnit.DefaultKPU.GetBuffer(data);
        }

        public override bool Equals(ITensor? other)
            => other is DataTensor<T> tensor && buffer == tensor.buffer;
    }
}

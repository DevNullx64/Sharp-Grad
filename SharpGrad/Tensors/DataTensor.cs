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

        protected DataTensor(string name, Shape shape, AcceleratorBuffer<T> buffer) : base(name, shape)
        {
            Shape = shape;
            this.buffer = buffer;
        }

        public DataTensor(string name, Shape shape)
            : this(name, shape, KernelProcessUnit.DefaultKPU.GetBuffer<T>(shape.Length)) { }

        public DataTensor(string name, Shape shape, T[] data)
            : this(name, shape, KernelProcessUnit.DefaultKPU.GetBuffer(data)) { }

        public override bool Equals(ITensor? other)
            => other is DataTensor<T> tensor && buffer == tensor.buffer;
    }
}

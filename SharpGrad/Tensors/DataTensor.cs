using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class DataTensor<T>(Shape shape) : Tensor<T>(shape)
        where T : unmanaged, INumber<T>
    {
        private readonly AcceleratorBuffer<T> data = new(shape.Size);
        internal override ArrayView1D<T, Stride1D.Dense> GetArrayView1D() => data.AcceleratorData.View;

        public override T this[params Index[] indices] 
        {
            get => data[Shape.GetFlattenIndex(indices)];
            set => data[Shape.GetFlattenIndex(indices)] = value;
        }

        public override T[] this[params Range[] ranges] 
        {
            get => throw new NotImplementedException();
            set => throw new NotImplementedException();
        }

        public DataTensor(params Dim[] dims) : this(new Shape(dims)) { }

    }
}

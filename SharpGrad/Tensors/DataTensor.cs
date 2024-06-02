using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class DataTensor<T>(Shape shape) : Tensor<T>(shape)
        where T : unmanaged, INumber<T>
    {
        private readonly AcceleratorBuffer<T> data = new(shape.Length);
        internal override ArrayView1D<T, Stride1D.Dense> GetArrayView1D() => data.AcceleratorData.View;

        public override T this[params Index[] indices] 
        {
            get => data[Shape.GetFlattenIndex(indices)];
            set => data[Shape.GetFlattenIndex(indices)] = value;
        }

        public override T[,] this[params Range[] ranges] 
        {
            get => throw new NotImplementedException();
            set => throw new NotImplementedException();
        }

        public DataTensor(params Dim[] dims) : this(new Shape(dims)) { }

    }

    public class DataTensor<T, TGrad>(Shape shape) : DataTensor<T>(shape), ITensorGrad<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        protected AcceleratorBuffer<TGrad> grad = new(shape.Length);
        public void AddGrad(AcceleratorBuffer<TGrad> grad) => throw new NotImplementedException();
        public void ApplyGrad(TGrad lr) => throw new NotImplementedException();
    }
}

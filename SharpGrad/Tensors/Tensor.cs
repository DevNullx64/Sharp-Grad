using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public abstract class Tensor<T>(Shape shape) : ITensor<T>
        where T : unmanaged, INumber<T>
    {
        public Shape Shape { get; } = shape;

        public abstract T this[params Index[] indices] { get;set; }
        public virtual T[] this[params Range[] ranges] 
        {
            get => throw new NotImplementedException(); 
            set => throw new NotImplementedException(); 
        }

        internal abstract ArrayView1D<T, Stride1D.Dense> GetArrayView1D();

        public static Tensor<T> operator +(Tensor<T> left, Tensor<T> right) => new TensorOp<T, AddOp<T>>(left, right);

        public static Tensor<T> operator -(Tensor<T> left, Tensor<T> right) => new TensorOp<T, SubOp<T>>(left, right);

        public static Tensor<T> operator *(Tensor<T> left, Tensor<T> right) => new TensorOp<T, MulOp<T>>(left, right);

        public static Tensor<T> operator /(Tensor<T> left, Tensor<T> right) => new TensorOp<T, DivOp<T>>(left, right);
    }
}

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
        public long Length => Shape.Length;

        public abstract T this[params Index[] indices] { get;set; }
        public virtual T[,] this[params Range[] ranges] 
        {
            get => throw new NotImplementedException(); 
            set => throw new NotImplementedException(); 
        }

        internal abstract ArrayView1D<T, Stride1D.Dense> GetArrayView1D();

        public static Tensor<T> operator +(Tensor<T> operand1, Tensor<T> operand2) => new TensorOp2<T, AddOp<T>>(operand1, operand2);

        public static Tensor<T> operator -(Tensor<T> operand1, Tensor<T> operand2) => new TensorOp2<T, SubOp<T>>(operand1, operand2);

        public static Tensor<T> operator -(Tensor<T> operand1) => new TensorOp1<T, NegOp<T>>(operand1);

        public static Tensor<T> operator *(Tensor<T> operand1, Tensor<T> operand2) => new TensorOp2<T, MulOp<T>>(operand1, operand2);

        public static Tensor<T> operator /(Tensor<T> operand1, Tensor<T> operand2) => new TensorOp2<T, DivOp<T>>(operand1, operand2);

        //public static Tensor<U> CastTo<U>(Tensor<T> operand1)
        //    where U : unmanaged, INumber<U>
        //    => new CastOp<T, U>(operand1);
    }
}

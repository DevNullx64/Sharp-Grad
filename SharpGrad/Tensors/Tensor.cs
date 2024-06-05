using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public abstract class Tensor<T>(Shape shape) : ITensor<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
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

        internal static AcceleratorBuffer<T> Pow(ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right)
        {
            AcceleratorBuffer<T> result = Acc.GetAcceleratorBuffer<T>(left.Length);
            Acc.Exec(PowOp<T>.Exec, left, right, result.AcceleratorData.View);
            return result;
        }

        public static AcceleratorBuffer<T> Pow(AcceleratorBuffer<T> arrayView1D, AcceleratorBuffer<T> exponent)
        {
            AcceleratorBuffer<T> result = Acc.GetAcceleratorBuffer<T>(arrayView1D.Length);
            Acc.Exec(PowOp<T>.Exec, arrayView1D.AcceleratorData, exponent.AcceleratorData, result.AcceleratorData);
            return result;
        }

        internal static AcceleratorBuffer<T> Log(ArrayView1D<T, Stride1D.Dense> arrayView1D)
        {
            AcceleratorBuffer<T> result = Acc.GetAcceleratorBuffer<T>(arrayView1D.Length);
            Acc.Exec(LogOp<T>.Exec, arrayView1D, result.AcceleratorData.View);
            return result;
        }

        public static Tensor<T> operator +(Tensor<T> operand1, Tensor<T> operand2) => new TensorOp2<T, AddOp<T>>(operand1, operand2);

        public static Tensor<T> operator -(Tensor<T> operand1, Tensor<T> operand2) => new TensorOp2<T, SubOp<T>>(operand1, operand2);

        public static Tensor<T> operator -(Tensor<T> operand1) => new TensorOp1<T, NegOp<T>>(operand1);

        public static Tensor<T> operator *(Tensor<T> operand1, Tensor<T> operand2) => new TensorOp2<T, MulOp<T>>(operand1, operand2);

        public static Tensor<T> operator /(Tensor<T> operand1, Tensor<T> operand2) => new TensorOp2<T, DivOp<T>>(operand1, operand2);

        public static Tensor<T> Pow(Tensor<T> left, Tensor<T> right) => new TensorOp2<T, PowOp<T>>(left, right);
    }
}

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    internal class PowOp<T> : IOperation11_2<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape ResultingShape(Shape operand1, Shape operand2) => operand1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Exec(T left, T right) => T.Pow(left, right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(left[idx], right[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> operand1, T operand2, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(operand1[idx], operand2);

        public static void Backward(TensorGrad<T> @this, TensorGrad<T> operand1, TensorGrad<T> operand2)
        {
            operand1.AddGrad(@this.Gradients * Tensor<T>.Pow(operand1.GetArrayView1D(), operand2.GetArrayView1D()) * Tensor<T>.Log(operand1.GetArrayView1D()));
            operand2.AddGrad(@this.Gradients * Tensor<T>.Pow(operand1.GetArrayView1D(), operand2.GetArrayView1D()) * Tensor<T>.Log(operand2.GetArrayView1D()));
        }

        public static void Backward(Index1D idx, ArrayView1D<T, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> operand2, ArrayView1D<T, Stride1D.Dense> grad1, ArrayView1D<T, Stride1D.Dense> grad2)
        {
            operand1[idx] += grad[idx] * T.Pow(operand1[idx], operand2[idx]) * T.Log(operand1[idx]);
            operand2[idx] += grad[idx] * T.Pow(operand1[idx], operand2[idx]) * T.Log(operand2[idx]);
        }
    }
}

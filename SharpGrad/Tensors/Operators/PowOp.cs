using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    internal class PowOp<T> : IOperation11_2<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        public static string Name => "^";

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape ResultingShape(Shape operand1, Shape operand2) => operand1.IsScalar ? operand2 : operand1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Exec(T left, T right) => T.Pow(left, right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<T, Stride1D.Dense> result)
        {
            result[idx] = right.Length == 1
                ? Exec(left[idx], right[0])
                : Exec(left[idx], right[idx]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> operand1, T operand2, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(operand1[idx], operand2);

        public static void Backward(TensorGrad<T> @this, Tensor<T> operand1, Tensor<T> operand2)
        {
            if (operand1 is TensorGrad<T> grad1 && operand2 is TensorData<T> op2)
                grad1.AddGrad(@this.Gradients * op2.data * AcceleratorBuffer<T>.Pow(grad1.data, op2.data - T.One));
            if (operand1 is TensorData<T> op1 && operand2 is TensorGrad<T> grad2)
                grad2.AddGrad(@this.Gradients * AcceleratorBuffer<T>.Pow(op1.data, grad2.data) * AcceleratorBuffer<T>.Log(op1.data));
        }

        public static void Backward(Index1D idx, ArrayView1D<T, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> operand2, ArrayView1D<T, Stride1D.Dense> grad1, ArrayView1D<T, Stride1D.Dense> grad2)
        {
            if (grad1.Length == 1)
            {
                // TODO : Implement scalar backward
            }
            else if (grad1.Length > 1)
                grad1[idx] += grad[idx] * operand2[idx] * T.Pow(operand1[idx], operand2[idx] - T.One);

            if (grad2.Length == 1)
            {
                // TODO : Implement scalar backward
            }
            else if (grad2.Length > 1)
                grad2[idx] += grad[idx] * T.Pow(operand1[idx], operand2[idx]) * T.Log(operand1[idx]);
        }
    }
}

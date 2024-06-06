using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    internal class MatMulOp<T> : IOperation11_2<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>, IMultiplyOperators<T, T, T>
    {
        public static string Name => "⋅";

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape ResultingShape(Shape operand1, Shape operand2){
            if(operand1.IsScalar || operand2.IsScalar)
                return operand1.IsScalar ? operand2 : operand1;
            if(operand1[^1] == operand2[0])
                return new Shape(operand1[0], operand2[1]);
            throw new NotSupportedException($"Incompatible shapes for matmul: {operand1} and {operand2}");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Exec(T left, T right) => left * right;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<T, Stride1D.Dense> result)
        {
            result[idx] = right.Length == 1
                    ? Exec(left[idx], right[0])
                    : Exec(left[idx], right[idx]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Backward(TensorGrad<T> @this, Tensor<T> operand1, Tensor<T> operand2)
        {
            if(operand1 is TensorGrad<T> grad1 && operand2 is TensorData<T> op2)
                grad1.AddGrad(@this.Gradients * op2.data);
            if(operand1 is TensorData<T> op1 && operand2 is TensorGrad<T> grad2)
                grad2.AddGrad(@this.Gradients * op1.data);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Backward(Index1D idx, ArrayView1D<T, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> operand2, ArrayView1D<T, Stride1D.Dense> grad1, ArrayView1D<T, Stride1D.Dense> grad2)
        {
        }
    }
}

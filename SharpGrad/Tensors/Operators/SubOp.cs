using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    internal class SubOp<T> : IOperation11_2<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>, ISubtractionOperators<T, T, T>
    {
        public static string Name => "-";

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape ResultingShape(Shape operand1, Shape operand2) => operand1.IsScalar ? operand2 : operand1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Exec(T left, T right) => left - right;

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
                grad1.AddGrad(@this.Gradients);
            if (operand1 is TensorData<T> op1 && operand2 is TensorGrad<T> grad2)
                grad2.AddGrad(-@this.Gradients);
        }

        public static void Backward(Index1D idx, ArrayView1D<T, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> operand2, ArrayView1D<T, Stride1D.Dense> grad1, ArrayView1D<T, Stride1D.Dense> grad2)
        {
            if (grad1.Length == 1)
            {
                // TODO : Implement scalar backward
            }
            else if (grad1.Length != 1)
                grad1[idx] += grad[idx];

            if (grad2.Length == 1)
            {
                // TODO : Implement scalar backward
            }
            else if (grad2.Length != 1)
                grad2[idx] -= grad[idx];
        }
    }
}

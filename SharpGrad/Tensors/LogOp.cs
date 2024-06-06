using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    internal class LogOp<T> : IOperation11_1<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape ResultingShape(Shape operand1) => operand1;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Exec(T left) => T.Log(left);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(left[idx]);
        public static void Backward(TensorGrad<T> @this, TensorGrad<T> operand1)
        {
            if (operand1 is TensorGrad<T> grad1)
                grad1.AddGrad(@this.Gradients / operand1.data);
        }

        public static void Backward(Index1D idx, ArrayView1D<T, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> grad1)
        {
            if (grad1.Length == 1)
            {
                // TODO : Implement scalar backward
            }
            else if (grad1.Length > 1)
            {
                grad1[idx] += grad[idx] / operand1[idx];
            }
        }
    }
}
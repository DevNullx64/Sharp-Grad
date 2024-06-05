using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    internal class NegOp<T> : IOperation11_1<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>, IUnaryNegationOperators<T, T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape ResultingShape(Shape operand1) => operand1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Exec(T operand1) => -operand1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(operand1[idx]);

        public static void Backward(TensorGrad<T> @this, TensorGrad<T> operand)
        {
            operand.AddGrad(-@this.Gradients);
        }

        public static void Backward(Index1D idx, ArrayView1D<T, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> grad1)
        {
            grad1[idx] += -grad[idx];
        }
    }
}

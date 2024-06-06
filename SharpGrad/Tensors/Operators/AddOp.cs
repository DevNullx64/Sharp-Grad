using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    internal interface IAddOpMixed<T1, T2, TTo, TGrad> : IOperation11_2_mixed<T1, T2, TTo, TGrad>
        where T1 : unmanaged, INumber<T1>, IAdditionOperators<T1, T2, TTo>
        where T2 : unmanaged, INumber<T2>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }

    internal class AddOp<T> : IOperation11_2<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>, IAdditionOperators<T, T, T>
    {
        public static string Name => "+";

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape ResultingShape(Shape operand1, Shape operand2) => operand1.IsScalar ? operand2 : operand1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Exec(T left, T right) => left + right;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<T, Stride1D.Dense> result)
        {
            result[idx] = (right.Length == 1)
                ? Exec(left[idx], right[0])
                : Exec(left[idx], right[idx]);
        }

        public static void Backward(TensorGrad<T> @this, Tensor<T> operand1, Tensor<T> operand2)
        {
            if (operand1 is TensorGrad<T> grad1)
                grad1.AddGrad(@this.Gradients);
            if(operand2 is TensorGrad<T> grad2)
                grad2.AddGrad(@this.Gradients);
        }

        public static void Backward(Index1D idx, ArrayView1D<T, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> operand2, ArrayView1D<T, Stride1D.Dense> grad1, ArrayView1D<T, Stride1D.Dense> grad2)
        {
            if (grad1.Length == 1)
            {
                // TODO : Implement scalar backward
            }
            else if (grad1.Length > 1)
                grad1[idx] += grad[idx];

            if (grad2.Length == 1)
            {
                // TODO : Implement scalar backward
            }
            else if(grad2.Length > 1)
                grad2[idx] += grad[idx];
        }
    }
}

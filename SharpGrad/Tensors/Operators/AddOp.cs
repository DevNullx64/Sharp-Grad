using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    internal interface IAddOpMixed<T1, T2, TTo>: IOperation11_2_mixed<T1, T2, TTo>
        where T1 : unmanaged, INumber<T1>, IAdditionOperators<T1, T2, TTo>
        where T2 : unmanaged, INumber<T2>
        where TTo : unmanaged, INumber<TTo>
    { }

    internal class AddOp<T> : IOperation11_2<T>
        where T : unmanaged, INumber<T>, IAdditionOperators<T, T, T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape ResultingShape(Shape operand1, Shape operand2) => operand1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Exec(T left, T right) => left + right;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(left[idx], right[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> operand1, T operand2, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(operand1[idx], operand2);
    }
}

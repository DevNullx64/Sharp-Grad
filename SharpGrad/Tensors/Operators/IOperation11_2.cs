using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public interface IOperation11_2_mixed<T1, T2, TTo, TGrad>
        where T1 : unmanaged, INumber<T1>
        where T2 : unmanaged, INumber<T2>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static Shape ResultingShape(Shape operand1, Shape operand2);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static TTo Exec(T1 operand1, T2 operand2);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Exec(Index1D idx, ArrayView1D<T1, Stride1D.Dense> operand1, ArrayView1D<T2, Stride1D.Dense> operand2, ArrayView1D<TTo, Stride1D.Dense> result);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Exec(Index1D idx, ArrayView1D<T1, Stride1D.Dense> operand1, T2 operand2, ArrayView1D<TTo, Stride1D.Dense> result);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Backward(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<T1, Stride1D.Dense> operand1, ArrayView1D<T2, Stride1D.Dense> operand2, ArrayView1D<TGrad, Stride1D.Dense> grad1, ArrayView1D<TGrad, Stride1D.Dense> grad2);
    }

    public interface IOperation11_2<T> : IOperation11_2_mixed<T, T, T, T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Backward(TensorGrad<T> @this, TensorGrad<T> operand1, TensorGrad<T> operand2);
    }
}

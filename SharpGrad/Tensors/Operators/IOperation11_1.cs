using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public interface IOperation11_1_mixed<T1, TTo, TGrad>
        where T1 : unmanaged, INumber<T1>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static Shape ResultingShape(Shape operand1);
        abstract static TTo Exec(T1 operand1);
        abstract static void Exec(Index1D idx, ArrayView1D<T1, Stride1D.Dense> operand1, ArrayView1D<TTo, Stride1D.Dense> result);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Backward(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<T1, Stride1D.Dense> operand1, ArrayView1D<TGrad, Stride1D.Dense> grad1);
    }

    public interface IOperation11_1<T> : IOperation11_1_mixed<T, T, T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Backward(TensorGrad<T> @this, TensorGrad<T> operand);
    }
}

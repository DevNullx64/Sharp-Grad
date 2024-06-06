using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public interface IOperationN1_1_mixed<T1, TTo, TGrad>
        where T1 : unmanaged, INumber<T1>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static Shape ResultingShape(Shape operand);
        abstract static TTo Exec(T1[] operand);
        abstract static void Exec(Index1D idx, ArrayView2D<T1, Stride2D.DenseY> operand, int width, ArrayView1D<TTo, Stride1D.Dense> result);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Backward(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView2D<T1, Stride2D.DenseY> operand1, ArrayView2D<TGrad, Stride2D.DenseY> grad1, long width);
    }
    public interface IOperationN1_1<T> : IOperationN1_1_mixed<T, T, T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Backward(TensorGrad<T> @this, TensorGrad<T> operand);
    }

    public interface IOperationN1_2_mixed<T1, T2, TTo, TGrad>
        where T1 : unmanaged, INumber<T1>
        where T2 : unmanaged, INumber<T2>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static Shape ResultingShape(Shape operand1, Shape operand2);
        abstract static TTo[] Exec(T1[,] operand1, T2[] operand2);
        abstract static void Exec(Index1D idx, ArrayView2D<T1, Stride2D.DenseY> operand1, int width1, ArrayView2D<T2, Stride2D.DenseY> operand2, int width2, ArrayView1D<TTo, Stride1D.Dense> result);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static void Backward(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView2D<T1, Stride2D.DenseY> operand1, ArrayView2D<T1, Stride2D.DenseY> grad1, long width1, ArrayView2D<T2, Stride2D.DenseY> operand2, ArrayView2D<T2, Stride2D.DenseY> grad2, long width2);
    }

}

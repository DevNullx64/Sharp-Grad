using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IOperation11_2_mixed<T1, T2, TTo>
        where T1 : unmanaged, INumber<T1>
        where T2 : unmanaged, INumber<T2>
        where TTo : unmanaged, INumber<TTo>
    {
        abstract static Shape ResultingShape(Shape operand1, Shape operand2);
        abstract static TTo Exec(T1 operand1, T2 operand2);
        abstract static void Exec(Index1D idx, ArrayView1D<T1, Stride1D.Dense> operand1, ArrayView1D<T2, Stride1D.Dense> operand2, ArrayView1D<TTo, Stride1D.Dense> result);
    }

    public interface IOperation11_2<T>: IOperation11_2_mixed<T, T, T>
        where T : unmanaged, INumber<T>
    { }
}

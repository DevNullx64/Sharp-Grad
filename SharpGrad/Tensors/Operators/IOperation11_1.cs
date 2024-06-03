using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IOperation11_1_mixed<T1, TTo>
        where T1 : unmanaged, INumber<T1>
        where TTo : unmanaged, INumber<TTo>
    {
        abstract static Shape ResultingShape(Shape operand1);
        abstract static TTo Exec(T1 operand1);
        abstract static void Exec(Index1D idx, ArrayView1D<T1, Stride1D.Dense> operand1, ArrayView1D<TTo, Stride1D.Dense> result);
    }

    public interface IOperation11_1<T> : IOperation11_1_mixed<T, T>
        where T : unmanaged, INumber<T>
    { }
}

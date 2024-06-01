using ILGPU.Runtime;
using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ICast<TFrom, TTo>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
    {
        abstract static Shape ResultingShape(Shape operand1);
        abstract static TTo Exec(TFrom operand1);
        abstract static void Exec(Index1D idx, ArrayView1D<TFrom, Stride1D.Dense> operand1, ArrayView1D<TTo, Stride1D.Dense> result);

    }

    internal class CastOp<TFrom, TTo> : ICast<TFrom, TTo>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
    {
        public static Shape ResultingShape(Shape operand1) => operand1;

        public static TTo Exec(TFrom operand1) => (TTo)(dynamic)operand1;
        public static void Exec(Index1D idx, ArrayView1D<TFrom, Stride1D.Dense> operand1, ArrayView1D<TTo, Stride1D.Dense> result)
            => result[idx] = Exec(operand1[idx]);
    }

}

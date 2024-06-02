using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class DivOp<T> : IOperation11_2<T>
        where T : unmanaged, INumber<T>, IDivisionOperators<T, T, T>
    {
        public static Shape ResultingShape(Shape operand1, Shape operand2) => operand1;
        public static T Exec(T left, T right) => left / right;
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(left[idx], right[idx]);

    }
}

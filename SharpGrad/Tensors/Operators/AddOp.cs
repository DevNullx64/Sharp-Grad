using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class AddOp<T> : IOperation2<T>
        where T : unmanaged, INumber<T>
    {
        public static Shape ResultingShape(Shape operand1, Shape operand2) => operand1;

        public AddOp()
        {
        }

        public static T Exec(T left, T right) => left + right;
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(left[idx], right[idx]);
    }
}

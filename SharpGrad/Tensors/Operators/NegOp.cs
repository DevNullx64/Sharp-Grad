using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal class NegOp<T> : IOperation1<T>
        where T : unmanaged, INumber<T>
    {
        public static Shape ResultingShape(Shape operand1) => operand1;

        public static T Exec(T operand1) => -operand1;
        public static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> result)
            => result[idx] = Exec(operand1[idx]);
    }
}

using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IOperationN1_1<T>
        where T : unmanaged, INumber<T>
    {
        abstract static Shape ResultingShape(Shape operand1);
        abstract static T Exec(T operand1);
        abstract static void Exec(Index1D idx, ArrayView1D<T, Stride1D.Dense> operand1, ArrayView1D<T, Stride1D.Dense> result);
    }
}

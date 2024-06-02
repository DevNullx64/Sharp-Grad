using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IOperationN1_1<T>
        where T : unmanaged, INumber<T>
    {
        abstract static Shape ResultingShape(Shape operand);
        abstract static T Exec(T[] operand);
        abstract static void Exec(Index1D idx, ArrayView2D<T, Stride2D.DenseY> operand, int width, ArrayView1D<T, Stride1D.Dense> result);
    }
}

using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IApplyOpOne<T> : IApplyOp
        where T : unmanaged, INumber<T>
    {
        abstract static T ApplyCpu(T left);
        abstract static void ApplyAccelerator(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> output);
    }
}
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IApplyOpOne<TFrom, TTo> : IApplyOp<TFrom, TTo>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
    {
        abstract static TTo ApplyCpu(TFrom left);
        abstract static void ApplyAccelerator(Index1D idx, ArrayView1D<TFrom, Stride1D.Dense> left, ArrayView1D<TTo, Stride1D.Dense> output);
    }
}
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IApplyOpTwo<TFrom, TTo> : IApplyOp<TFrom, TTo>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
    {
        abstract static Shape ResultShape(Shape left, Shape right);

        abstract static TTo ApplyCpu(TFrom left, TFrom right);
        abstract static void ApplyAccelerator(Index1D idx, ArrayView1D<TFrom, Stride1D.Dense> left, ArrayView1D<TFrom, Stride1D.Dense> right, ArrayView1D<TTo, Stride1D.Dense> output);
    }
}
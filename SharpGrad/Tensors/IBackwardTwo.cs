using ILGPU.Runtime;
using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardTwo<TFrom, TTo, TGrad> : IApplyOpTwo<TFrom, TTo>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, TFrom left, TFrom right);
        abstract static void BackwardAccelerator(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<TFrom, Stride1D.Dense> left, ArrayView1D<TFrom, Stride1D.Dense> right, ArrayView1D<TGrad, Stride1D.Dense> leftGrad, ArrayView1D<TGrad, Stride1D.Dense> rightGrad);
    }
}
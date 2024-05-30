using ILGPU.Runtime;
using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardOne<TFrom, TTo, TGrad> : IApplyOpOne<TFrom, TTo>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static TGrad BackwardCpu(TGrad grad, TFrom left);
        abstract static void BackwardAccelerator(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<TFrom, Stride1D.Dense> left, ArrayView1D<TGrad, Stride1D.Dense> leftGrad);
    }


}
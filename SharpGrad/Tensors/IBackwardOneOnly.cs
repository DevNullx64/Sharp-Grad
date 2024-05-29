using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardOneOnly<T, TGrad> : IBackward
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static TGrad BackwardCpu(TGrad grad, T left);
        abstract static void BackwardAccelerator(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<TGrad, Stride1D.Dense> leftGrad);
    }


}
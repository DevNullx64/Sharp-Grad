using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardTwoOnly<T, TGrad> : IBackward
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, T left, T right);
        abstract static void BackwardAccelerator(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<TGrad, Stride1D.Dense> leftGrad, ArrayView1D<TGrad, Stride1D.Dense> rightGrad);
    }


}
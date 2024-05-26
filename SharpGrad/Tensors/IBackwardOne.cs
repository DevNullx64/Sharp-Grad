using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardOne<TType, TGrad> : IApplyOpOne<TType>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static TGrad BackwardCpu(TGrad grad, TType left);
        abstract static void BackwardGpu(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TGrad> leftGrad);
    }
}
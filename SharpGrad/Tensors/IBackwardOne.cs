using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardOne<TType> : IApplyOpOne<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        abstract static TType BackwardCpu(TType grad, TType left);
        abstract static void BackwardGpu(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> leftGrad);
    }
}
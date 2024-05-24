using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardOne<TType> : IApplyOp<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        abstract static TType Backward(TType grad, TType left);
        abstract static void Backward(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> leftGrad);
    }
}
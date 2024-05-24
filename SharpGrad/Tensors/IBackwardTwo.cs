using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardTwo<TType> : IApplyOp<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        abstract static (TType Left, TType Right) Backward(TType grad, TType left, TType right);
        abstract static void Backward(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> leftGrad, ArrayView<TType> rightGrad);
    }
}
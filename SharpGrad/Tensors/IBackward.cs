using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackward { }

    public interface IBackwardOne<TType, TGrad> : IApplyOpOne<TType>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static TGrad BackwardCpu(TGrad grad, TType left);
        abstract static void BackwardGpu(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TGrad> leftGrad);
    }

    public interface IBackwardTwo<TType, TGrad> : IApplyOpTwo<TType>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, TType left, TType right);
        abstract static void BackwardGpu(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TGrad> leftGrad, ArrayView<TGrad> rightGrad);
    }

}
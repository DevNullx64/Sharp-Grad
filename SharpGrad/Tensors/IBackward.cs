using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackward { }


    public interface IBackwardable { }
    

    public interface IBackwardOneOnly<TType, TGrad> : IBackward
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static TGrad BackwardCpu(TGrad grad, TType left);
        abstract static void BackwardAccelerator(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TGrad> leftGrad);
    }
    public interface IBackwardOne<TType, TGrad> : IApplyOpOne<TType>, IBackwardOneOnly<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }

    public interface IBackwardableOne<TType, TGrad, TOp> : IBackwardable
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
        where TOp : IBackwardOne<TType, TGrad>
    {
        public void BackwardCpu();
    }


    public interface IBackwardTwoOnly<TType, TGrad> : IBackward
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        abstract static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, TType left, TType right);
        abstract static void BackwardAccelerator(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TGrad> leftGrad, ArrayView<TGrad> rightGrad);
    }

    public interface IBackwardTwo<TType, TGrad> : IApplyOpTwo<TType>, IBackwardTwoOnly<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }

    public interface IBackwardableTwo<TType, TGrad, TOp> : IBackwardable
    where TType : unmanaged, INumber<TType>
    where TGrad : unmanaged, IFloatingPoint<TGrad>
    where TOp : IBackwardTwo<TType, TGrad>
    {
        public void Backward();
    }


}
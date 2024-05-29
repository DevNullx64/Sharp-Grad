using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardTwo<T, TGrad> : IApplyOpTwo<T>, IBackwardTwoOnly<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }


}
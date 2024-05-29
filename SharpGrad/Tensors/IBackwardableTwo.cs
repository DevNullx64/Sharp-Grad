using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardableTwo<T, TGrad, TOp> : IBackwardable
    where T : unmanaged, INumber<T>
    where TGrad : unmanaged, IFloatingPoint<TGrad>
    where TOp : IBackwardTwo<T, TGrad>
    {
        public void Backward();
    }


}
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardableOne<T, TGrad, TOp> : IBackwardable
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
        where TOp : IBackwardOne<T, TGrad>
    {
        public void BackwardCpu();
    }


}
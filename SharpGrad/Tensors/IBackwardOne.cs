using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackwardOne<T, TGrad> : IApplyOpOne<T>, IBackwardOneOnly<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }


}
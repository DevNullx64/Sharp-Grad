using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IBackward<TFrom, TTo, TGrad>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }
}
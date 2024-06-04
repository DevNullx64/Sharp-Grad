using System.Numerics;

namespace SharpGrad.Tensors
{

    public interface ITensorOpBackward<T, TGrad> : ITensorGrad<T, TGrad>, IGradient<T, TGrad>, IOpBackward<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }

    public interface ITensorOpBackward<T> : ITensorGrad<T>, IGradient<T>, IOpBackward<T>
    where T : unmanaged, IFloatingPoint<T>
    { }
}

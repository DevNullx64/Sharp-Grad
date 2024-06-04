using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorGrad<T, TGrad>: ITensor<T>, IGradient<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }

}

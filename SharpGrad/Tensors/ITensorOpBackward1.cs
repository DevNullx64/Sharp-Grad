using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorOpBackward<T> : ITensorOpBackward<T, T>
        where T : unmanaged, IFloatingPoint<T>
    { }

}

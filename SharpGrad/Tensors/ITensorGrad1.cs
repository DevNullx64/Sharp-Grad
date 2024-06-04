using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorGrad<T> : ITensorGrad<T, T>
        where T : unmanaged, IFloatingPoint<T>
    { }

}

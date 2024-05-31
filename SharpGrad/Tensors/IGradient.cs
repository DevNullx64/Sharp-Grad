using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IGradient<T>
        where T : unmanaged, IFloatingPoint<T>
    {
        void SetToZero();
        void SetToOne();
        void AddGradient(T value);
    }
}

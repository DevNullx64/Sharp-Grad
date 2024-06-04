using System.Numerics;
using SharpGrad.Memory;

namespace SharpGrad.Tensors
{
    public interface IGradient<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public void AddGrad(AcceleratorBuffer<T> grad);
        public void ApplyGrad(TGrad lr);
    }
    public interface IGradient<T> : IGradient<T, T>
        where T : unmanaged, IFloatingPoint<T>
    { }

}

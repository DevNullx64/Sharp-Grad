using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorGrad<T, TGrad>: ITensor<T>, IGradient<T, TGrad>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }
    public interface ITensorGrad<T> : ITensor<T>, IGradient<T>
    where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    { }
}

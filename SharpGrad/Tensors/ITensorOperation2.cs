using SharpGrad.Tensors.Operators;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal interface ITensorOperation2<T> : ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        Tensor<T> Left { get; }
        Tensor<T> Right { get; }
    }

    internal interface ITensorOperation2<T, TOp> : ITensorOperation2<T>
    where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    where TOp : IExecBinary<T, T, T>
    { }
}

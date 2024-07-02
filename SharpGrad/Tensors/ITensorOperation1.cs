using SharpGrad.Tensors.Operators;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal interface ITensorOperation1<T> : ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        Tensor<T> Operand { get; }
    }

    internal interface ITensorOperation1<T, TOp> : ITensorOperation1<T>
    where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    where TOp : IExecutor1<T, T>
    { }
}

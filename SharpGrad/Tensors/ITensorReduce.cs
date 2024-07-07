using SharpGrad.Tensors.Operators;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal interface ITensorReduce<T> : ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        Tensor<T> Operand { get; }
    }

    internal interface ITensorReduce<T, TOp> : ITensorReduce<T>
    where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    where TOp : IExecutor2<T, T, T>
    { }
}

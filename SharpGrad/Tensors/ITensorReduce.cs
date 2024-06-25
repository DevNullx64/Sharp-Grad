using SharpGrad.Tensors.Operators;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal interface ITensorReduce<T> : ITensorOperation1<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    { }

    internal interface ITensorReduce<T, TOp> : ITensorReduce<T>
    where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    where TOp : IAggregator<T, TOp>
    { }
}

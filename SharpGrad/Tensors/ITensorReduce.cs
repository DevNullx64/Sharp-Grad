using SharpGrad.Tensors.Operators;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal interface ITensorReduce<T> : ITensorOperation1<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    { }

    internal interface ITensorReduce<T, TOp> : ITensorReduce<T>
    where T : unmanaged, INumber<T>, IPowerFunctions<T>
    where TOp : IAggregator<T, TOp>
    { }
}

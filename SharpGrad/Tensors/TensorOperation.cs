using SharpGrad.Tensors.KPU;
using SharpGrad.Tensors.Operators;
using System.Numerics;

namespace SharpGrad.Tensors
{
    internal abstract class TensorOperation<T, TOp>(Shape shape) : Tensor<T>(TOp.Symbol, shape), ITensorOperation<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        where TOp : IExec
    {
        public OpCode OpCode => TOp.OpCode;
    }
}

using System.Collections.Generic;
using System.Numerics;
using SharpGrad.Tensors.KPU;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// Base interface for tensor operations.
    /// </summary>
    public interface ITensorOperation : ITensor
    {
        /// <summary>
        /// The operation code of the <see cref="ITensorOperation"/>
        /// </summary>
        OpCode OpCode { get; }
    }

    public interface ITensorOperation<T> : ITensorOperation, ITensor<T>
    where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        void Backward();
    }
}

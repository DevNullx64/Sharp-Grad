using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorOperation : ITensor
    {
        OpCode OpCode { get; }
    }

    public interface ITensorOperation<T> : ITensorOperation, ITensor<T>
    where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        void Backward();
    }
}

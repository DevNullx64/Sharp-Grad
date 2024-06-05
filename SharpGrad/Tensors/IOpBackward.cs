using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IOpBackward<T, TGrad>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public void Backward();
        public void DepthFirstSearch(HashSet<Tensor<T>> visited, Stack<Tensor<T>> stack);
    }

    public interface IOpBackward<T> : IOpBackward<T, T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    { }

}

using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorOpBackward<T, TGrad>: ITensorGrad<T, TGrad>, IGradient<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public void Backward();
        public void DepthFirstSearch(HashSet<Tensor<T>> visited, Stack<Tensor<T>> stack);
    }

}

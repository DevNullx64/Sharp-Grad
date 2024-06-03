using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensor<T>
        where T : unmanaged, INumber<T>
        
    {
        public Shape Shape { get; }
        public long Length { get; }

        public T this[params Index[] indices] { get; set; }
        public T[,] this[params Range[] ranges] { get; set; }
    }

    public interface ITensorGrad<T, TGrad>: ITensor<T>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public void AddGrad(AcceleratorBuffer<T> grad);
        public void ApplyGrad(TGrad lr);
    }

    public interface ITensorGrad<T> : ITensorGrad<T, T>
        where T : unmanaged, IFloatingPoint<T>
    { }


    public interface ITensorOpBackward<T, TGrad>: ITensorGrad<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public void DepthFirstSearch(HashSet<Tensor<T>>? visited, Stack<Tensor<T>>? stack);
        public void Backward();
    }
}

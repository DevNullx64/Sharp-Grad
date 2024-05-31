using System.Numerics;

namespace SharpGrad.Tensors
{

    public interface ITensor<TSelf, T> : ITensorBase<T>,
        IAdditionOperators<TSelf, Tensor<T>, Tensor<T>>,
        ISubtractionOperators<TSelf, Tensor<T>, Tensor<T>>,
        IMultiplyOperators<TSelf, Tensor<T>, Tensor<T>>,
        IDivisionOperators<TSelf, Tensor<T>, Tensor<T>>
        where TSelf : ITensor<TSelf, T>
        where T : unmanaged, INumber<T>
    { }

    public interface ITensor<TSelf, T, TGrad> : ITensor<TSelf, T>, IGradient<TGrad>,
        IAdditionOperators<TSelf, Tensor<T>, Tensor<T, TGrad>>,
        IAdditionOperators<TSelf, Tensor<T, TGrad>, Tensor<T, TGrad>>,
        ISubtractionOperators<TSelf, Tensor<T>, Tensor<T, TGrad>>,
        ISubtractionOperators<TSelf, Tensor<T, TGrad>, Tensor<T, TGrad>>,
        IMultiplyOperators<TSelf, Tensor<T>, Tensor<T, TGrad>>,
        IMultiplyOperators<TSelf, Tensor<T, TGrad>, Tensor<T, TGrad>>,
        IDivisionOperators<TSelf, Tensor<T>, Tensor<T, TGrad>>,
        IDivisionOperators<TSelf, Tensor<T, TGrad>, Tensor<T, TGrad>>
        where TSelf : ITensor<TSelf, T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }
}

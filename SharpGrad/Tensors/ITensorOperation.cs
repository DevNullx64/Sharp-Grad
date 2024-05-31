using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorOperation<TSelf, TFrom, TTo> : ITensorBase<TTo>,
        IAdditionOperators<TSelf, Tensor<TFrom>, Tensor<TTo>>,
        ISubtractionOperators<TSelf, Tensor<TFrom>, Tensor<TTo>>,
        IMultiplyOperators<TSelf, Tensor<TFrom>, Tensor<TTo>>,
        IDivisionOperators<TSelf, Tensor<TFrom>, Tensor<TTo>>
        where TSelf : ITensorOperation<TSelf, TFrom, TTo>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
    { }

    public interface ITensorOperation<TSelf, TFrom, TTo, TGrad> : ITensorOperation<TSelf, TFrom, TTo>, IGradient<TGrad>,
    IAdditionOperators<TSelf, Tensor<TFrom>, Tensor<TTo, TGrad>>,
    IAdditionOperators<TSelf, Tensor<TTo, TGrad>, Tensor<TTo, TGrad>>,
    ISubtractionOperators<TSelf, Tensor<TFrom>, Tensor<TTo, TGrad>>,
    ISubtractionOperators<TSelf, Tensor<TTo, TGrad>, Tensor<TTo, TGrad>>,
    IMultiplyOperators<TSelf, Tensor<TFrom>, Tensor<TTo, TGrad>>,
    IMultiplyOperators<TSelf, Tensor<TTo, TGrad>, Tensor<TTo, TGrad>>,
    IDivisionOperators<TSelf, Tensor<TFrom>, Tensor<TTo, TGrad>>,
    IDivisionOperators<TSelf, Tensor<TTo, TGrad>, Tensor<TTo, TGrad>>
    where TSelf : ITensorOperation<TSelf, TFrom, TTo, TGrad>
    where TFrom : unmanaged, INumber<TFrom>
    where TTo : unmanaged, INumber<TTo>
    where TGrad : unmanaged, IFloatingPoint<TGrad>

    { }
}

using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensor {
        Shape Shape { get; }
        bool IsGradients { get; set; }

        void ResetGradients();
    }

    public interface ITensor<TSelf, TType> : ITensor,
        IAdditionOperators<TSelf, TSelf, Tensor<TType, float>>,
        ISubtractionOperators<TSelf, TSelf, Tensor<TType, float>>,
        IMultiplyOperators<TSelf, TSelf, Tensor<TType, float>>,
        IDivisionOperators<TSelf, TSelf, Tensor<TType, float>>
        where TSelf : ITensor<TSelf, TType>
        where TType : unmanaged, INumber<TType>
    {
        TType this[params int[] indices] { get; set; }
    }
    public interface ITensor<TSelf, TType, TGrad> : ITensor,
        IAdditionOperators<TSelf, TSelf, Tensor<TType, TGrad>>,
        ISubtractionOperators<TSelf, TSelf, Tensor<TType, TGrad>>,
        IMultiplyOperators<TSelf, TSelf, Tensor<TType, TGrad>>,
        IDivisionOperators<TSelf, TSelf, Tensor<TType, TGrad>>
        where TSelf : ITensor<TSelf, TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        TType this[params int[] indices] { get; set; }
    }
}

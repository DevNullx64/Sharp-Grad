using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensor {
        Shape Shape { get; }
    }

    public interface ITensor<TSelf, TType> : ITensor,
        IAdditionOperators<TSelf, TSelf, TensorBase<TType, float>>,
        ISubtractionOperators<TSelf, TSelf, TensorBase<TType, float>>,
        IMultiplyOperators<TSelf, TSelf, TensorBase<TType, float>>,
        IDivisionOperators<TSelf, TSelf, TensorBase<TType, float>>
        where TSelf : ITensor<TSelf, TType>
        where TType : unmanaged, INumber<TType>
    {
        TType this[params int[] indices] { get; set; }
    }
    public interface ITensor<TSelf, TType, TGrad> : ITensor,
        IAdditionOperators<TSelf, TSelf, TensorBase<TType, TGrad>>,
        ISubtractionOperators<TSelf, TSelf, TensorBase<TType, TGrad>>,
        IMultiplyOperators<TSelf, TSelf, TensorBase<TType, TGrad>>,
        IDivisionOperators<TSelf, TSelf, TensorBase<TType, TGrad>>
        where TSelf : ITensor<TSelf, TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        TType this[params int[] indices] { get; set; }
    }
}

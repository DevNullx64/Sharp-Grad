using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensor {
        Shape Shape { get; }
    }

    public interface ITensor<TSelf, TType> : ITensor,
        IAdditionOperators<TSelf, TSelf, TSelf>,
        ISubtractionOperators<TSelf, TSelf, TSelf>,
        IMultiplyOperators<TSelf, TSelf, TSelf>,
        IDivisionOperators<TSelf, TSelf, TSelf>
        where TSelf : ITensor<TSelf, TType>
    {
        TType this[params int[] indices] { get; set; }
    }
}

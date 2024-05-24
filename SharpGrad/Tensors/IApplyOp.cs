using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IApplyOp<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        abstract static TType Apply(TType left, TType right);
    }
}
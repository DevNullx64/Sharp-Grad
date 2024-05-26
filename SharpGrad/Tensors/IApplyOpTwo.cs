using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IApplyOp { }

    public interface IApplyOpOne<TType> : IApplyOp
        where TType : unmanaged, IFloatingPoint<TType>
    {
        abstract static TType ApplyCpu(TType left);
        abstract static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> output);
    }

    public interface IApplyOpTwo<TType> : IApplyOp
        where TType : unmanaged, IFloatingPoint<TType>
    {
        abstract static TType ApplyCpu(TType left, TType right);
        abstract static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output);
    }
}
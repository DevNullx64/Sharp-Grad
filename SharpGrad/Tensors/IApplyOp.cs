using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface IApplyOp { }

    public interface IApplyOpOne<T> : IApplyOp
        where T : unmanaged, INumber<T>
    {
        abstract static T ApplyCpu(T left);
        abstract static void ApplyGpu(Index1D idx, ArrayView<T> left, ArrayView<T> output);
    }

    public interface IApplyOpTwo<T> : IApplyOp
        where T : unmanaged, INumber<T>
    {
        abstract static T ApplyCpu(T left, T right);
        abstract static void ApplyGpu(Index1D idx, ArrayView<T> left, ArrayView<T> right, ArrayView<T> output);
    }
}
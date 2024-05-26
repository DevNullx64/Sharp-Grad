using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct SubOp<TType> : IBackwardTwo<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static TType ApplyCpu(TType left, TType right)
            => left - right;

        public static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx], right[idx]);

        public static (TType Left, TType Right) BackwardCpu(TType grad, TType left, TType right)
            => (grad, -grad);

        public static void Apply(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx], right[idx]);

        public static void BackwardGpu(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> leftGrad, ArrayView<TType> rightGrad)
        {
            var (l, r) = SubOp<TType>.BackwardCpu(grad[idx], right[idx], left[idx]);
            leftGrad[idx] += l;
            rightGrad[idx] += r;
        }
    }
}
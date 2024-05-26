using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct NegOp<TType> : IBackwardOne<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static TType ApplyCpu(TType left) => -left;

        public static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> output) => output[idx] = ApplyCpu(left[idx]);

        public static TType BackwardCpu(TType grad, TType left) => -grad;

        public static void BackwardGpu(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> leftGrad) => leftGrad[idx] += BackwardCpu(grad[idx], left[idx]);
    }

    public struct ReLUOp<TType> : IBackwardOne<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static TType ApplyCpu(TType left)
            => left > TType.Zero ? left : TType.Zero;

         public static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx]);

        public static TType BackwardCpu(TType grad, TType left) => left > TType.Zero ? grad : TType.Zero;

        public static void Apply(Index1D idx, ArrayView<TType> left, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx]);

        public static void BackwardGpu(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> leftGrad)
        {
            var l = ReLUOp<TType>.BackwardCpu(grad[idx], left[idx]);
            leftGrad[idx] += l;
        }
    }
}
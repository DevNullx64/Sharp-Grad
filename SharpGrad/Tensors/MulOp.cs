using ILGPU;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    public struct MulOp<TType, TGrad> : IBackwardTwo<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static TType ApplyCpu(TType left, TType right)
            => left * right;

        public static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx], right[idx]);

        public static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, TType left, TType right)
            => (grad * TGrad.CreateChecked(right), grad * TGrad.CreateChecked(left));

        public static void Apply(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx], right[idx]);

        public static void BackwardGpu(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TGrad> leftGrad, ArrayView<TGrad> rightGrad)
        {
            var (l, r) = MulOp<TType, TGrad>.BackwardCpu(grad[idx], right[idx], left[idx]);
            leftGrad[idx] += l;
            rightGrad[idx] += r;
        }
    }
}
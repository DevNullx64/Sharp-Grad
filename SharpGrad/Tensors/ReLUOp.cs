using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct ReLUOp<TType, TGrad> : IBackwardOne<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static TType ApplyCpu(TType left)
            => left > TType.Zero ? left : TType.Zero;

         public static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx]);

        public static TGrad BackwardCpu(TGrad grad, TType left) => left > TType.Zero ? grad : TGrad.Zero;

        public static void BackwardGpu(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TGrad> leftGrad)
        {
            var l = ReLUOp<TType, TGrad>.BackwardCpu(grad[idx], left[idx]);
            leftGrad[idx] += l;
        }
    }
}
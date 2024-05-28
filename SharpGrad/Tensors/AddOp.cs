using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct AddOp<TType, TGrad> : IBackwardTwo<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static TType ApplyCpu(TType left, TType right) => left + right;

        public static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx], right[idx]);

        public static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, TType left, TType right) => (grad, grad);

        public static void BackwardAccelerator(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TGrad> leftGrad, ArrayView<TGrad> rightGrad)
        {
            var (l, r) = AddOp<TType, TGrad>.BackwardCpu(grad[idx], right[idx], left[idx]);
            leftGrad[idx] += l;
            rightGrad[idx] += r;
        }
    }
}
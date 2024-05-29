using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct ReLUOp<TType, TGrad> : IBackwardOne<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static TType ApplyCpu(TType left)
            => left > TType.Zero ? left : TType.Zero;

         public static void ApplyAccelerator(Index1D idx, ArrayView1D<TType, Stride1D.Dense> left, ArrayView1D<TType, Stride1D.Dense> output)
            => output[idx] = ApplyCpu(left[idx]);

        public static TGrad BackwardCpu(TGrad grad, TType left) => left > TType.Zero ? grad : TGrad.Zero;

        public static void BackwardAccelerator(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<TType, Stride1D.Dense> left, ArrayView1D<TGrad, Stride1D.Dense> leftGrad)
        {
            var l = ReLUOp<TType, TGrad>.BackwardCpu(grad[idx], left[idx]);
            leftGrad[idx] += l;
        }
    }
}
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct ReLUOp<T, TGrad> : IBackwardOne<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static T ApplyCpu(T left)
            => left > T.Zero ? left : T.Zero;

         public static void ApplyAccelerator(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> output)
            => output[idx] = ApplyCpu(left[idx]);

        public static TGrad BackwardCpu(TGrad grad, T left) => left > T.Zero ? grad : TGrad.Zero;

        public static void BackwardAccelerator(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<TGrad, Stride1D.Dense> leftGrad)
        {
            var l = ReLUOp<T, TGrad>.BackwardCpu(grad[idx], left[idx]);
            leftGrad[idx] += l;
        }
    }
}
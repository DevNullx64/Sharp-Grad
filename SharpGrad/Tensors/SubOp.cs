using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct SubOp<TType, TGrad> : IBackwardTwo<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static TType ApplyCpu(TType left, TType right)
            => left - right;

        public static void ApplyAccelerator(Index1D idx, ArrayView1D<TType, Stride1D.Dense> left, ArrayView1D<TType, Stride1D.Dense> right, ArrayView1D<TType, Stride1D.Dense> output)
            => output[idx] = ApplyCpu(left[idx], right[idx]);

        public static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, TType left, TType right)
            => (grad, -grad);

        public static void BackwardAccelerator(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<TType, Stride1D.Dense> left, ArrayView1D<TType, Stride1D.Dense> right, ArrayView1D<TGrad, Stride1D.Dense> leftGrad, ArrayView1D<TGrad, Stride1D.Dense> rightGrad)
        {
            var (l, r) = SubOp<TType, TGrad>.BackwardCpu(grad[idx], right[idx], left[idx]);
            leftGrad[idx] += l;
            rightGrad[idx] += r;
        }

        public static Shape ResultShape(Shape left, Shape right)
        {
            if (left == right)
                return left;
            if (left.IsScalar)
                return right;
            if (right.IsScalar)
                return left;
            throw new ArgumentException($"Expected shapes {left} and {right} to be equal or one of them to be scalar");
        }
    }
}
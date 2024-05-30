using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class SubOp<TFrom, TTo> : IApplyOpTwo<TFrom, TTo>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
    {
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

        public static TTo ApplyCpu(TFrom left, TFrom right)
            => TTo.CreateTruncating(left - right);

        public static void ApplyAccelerator(
            Index1D idx,
            ArrayView1D<TFrom, Stride1D.Dense> left,
            ArrayView1D<TFrom, Stride1D.Dense> right,
            ArrayView1D<TTo, Stride1D.Dense> output)
            => output[idx] = ApplyCpu(left[idx], right[idx]);
    }

    public class SubOp<TFrom, TTo, TGrad> : SubOp<TFrom, TTo>, IBackwardTwo<TFrom, TTo, TGrad>
        where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, TFrom left, TFrom right)
            => (grad, -grad);

        public static void BackwardAccelerator(
            Index1D idx,
            ArrayView1D<TGrad, Stride1D.Dense> grad,
            ArrayView1D<TFrom, Stride1D.Dense> left,
            ArrayView1D<TFrom, Stride1D.Dense> right,
            ArrayView1D<TGrad, Stride1D.Dense> leftGrad,
            ArrayView1D<TGrad, Stride1D.Dense> rightGrad)
        {
            var (l, r) = SubOp<TFrom, TTo, TGrad>.BackwardCpu(grad[idx], right[idx], left[idx]);
            leftGrad[idx] += l;
            rightGrad[idx] += r;
        }
    }
}
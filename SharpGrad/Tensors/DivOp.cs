using ILGPU;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct DivOp<TType, TGrad> : IBackwardTwo<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static TType ApplyCpu(TType left, TType right)
            => left / right;

        public static void ApplyAccelerator(Index1D idx, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[idx] = ApplyCpu(left[idx], right[idx]);

        public static (TGrad Left, TGrad Right) BackwardCpu(TGrad grad, TType left, TType right)
        {
            TGrad l = TGrad.CreateChecked(left);
            TGrad r = TGrad.CreateChecked(right);
            return (grad / r, -grad * l / (r * r));
        }

        public static void BackwardAccelerator(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TGrad> leftGrad, ArrayView<TGrad> rightGrad)
        {
            var (l, r) = DivOp<TType, TGrad>.BackwardCpu(grad[idx], right[idx], left[idx]);
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
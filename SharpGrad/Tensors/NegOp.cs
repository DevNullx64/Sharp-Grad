using ILGPU;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public struct NegOp<TType> : IBackwardOne<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static TType Apply(TType left, TType right) => -left;

        public static TType Backward(TType grad, TType left) => -grad;

        public static void Apply(Index1D idx, ArrayView<TType> left, ArrayView<TType> output)
            => output[idx] = Apply(left[idx], default);
        public static void Backward(Index1D idx, ArrayView<TType> grad, ArrayView<TType> left, ArrayView<TType> leftGrad)
        {
            var l = NegOp<TType>.Backward(grad[idx], left[idx]);
            leftGrad[idx] += l;
        }
    }
}
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    public class NegOp<TFrom, TTo> : IApplyOpOne<TFrom, TTo>
        where TFrom : unmanaged, IFloatingPoint<TFrom>
        where TTo : unmanaged, IFloatingPoint<TTo>
    {
        public static Shape ResultShape(Shape left) => left;

        public static TTo ApplyCpu(TFrom left) => TTo.CreateTruncating(-left);

        public static void ApplyAccelerator(Index1D idx,
            ArrayView1D<TFrom, Stride1D.Dense> left,
            ArrayView1D<TTo, Stride1D.Dense> output)
            => output[idx] = ApplyCpu(left[idx]);
    }

    public class NegOp<TFrom, TTo, TGrad> : NegOp<TFrom, TTo>, IBackwardOne<TFrom, TTo, TGrad>
        where TFrom : unmanaged, IFloatingPoint<TFrom>
        where TTo : unmanaged, IFloatingPoint<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        /// <summary>
        /// Calculates the gradients of the negation operator.
        /// </summary>
        /// <param name="grad">Gradients to backpropagate.</param>
        /// <param name="left">Operands.</param>
        /// <returns>Gradients of the negations operator.</returns>
        public static TGrad BackwardCpu(TGrad grad, TFrom left) => -grad;

        /// <summary>
        /// Calculates the gradients of the negation operator on the <see cref="Accelerator"/>.
        /// </summary>
        /// <param name="idx"><see cref="Accelerator"/> index.</param>
        /// <param name="grad">Gradients to backpropagate.</param>
        /// <param name="left">Operands.</param>
        /// <param name="leftGrad">Gradients of the negations operator.</param>
        public static void BackwardAccelerator(
            Index1D idx,
            ArrayView1D<TGrad, Stride1D.Dense> grad,
            ArrayView1D<TFrom, Stride1D.Dense> left,
            ArrayView1D<TGrad, Stride1D.Dense> leftGrad)
            => leftGrad[idx] += BackwardCpu(grad[idx], left[idx]);
    }
}
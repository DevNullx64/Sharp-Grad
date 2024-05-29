using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// Negation operator.
    /// </summary>
    /// <typeparam name="T">Operand type.</typeparam>
    /// <typeparam name="TGrad">Type used for gradient calculations.</typeparam>
    public struct NegOp<T, TGrad> : IBackwardOne<T, TGrad>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        /// <summary>
        /// Negates the operand.
        /// </summary>
        /// <param name="left">Operand.</param>
        public static T ApplyCpu(T left) => -left;

        /// <summary>
        /// Negates the operand on the <see cref="Accelerator"/>
        /// </summary>
        /// <param name="idx"><see cref="Accelerator"/> index.</param>
        /// <param name="left">Operand.</param>
        /// <param name="output">Result of the negation.</param>
        public static void ApplyAccelerator(Index1D idx, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> output) => output[idx] = ApplyCpu(left[idx]);

        /// <summary>
        /// Calculates the gradients of the negation operator.
        /// </summary>
        /// <param name="grad">Gradients to backpropagate.</param>
        /// <param name="left">Operands.</param>
        /// <returns>Gradients of the negations operator.</returns>
        public static TGrad BackwardCpu(TGrad grad, T left) => -grad;

        /// <summary>
        /// Calculates the gradients of the negation operator on the <see cref="Accelerator"/>.
        /// </summary>
        /// <param name="idx"><see cref="Accelerator"/> index.</param>
        /// <param name="grad">Gradients to backpropagate.</param>
        /// <param name="left">Operands.</param>
        /// <param name="leftGrad">Gradients of the negations operator.</param>
        public static void BackwardAccelerator(Index1D idx, ArrayView1D<TGrad, Stride1D.Dense> grad, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<TGrad, Stride1D.Dense> leftGrad) => leftGrad[idx] += BackwardCpu(grad[idx], left[idx]);
    }
}
using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// Negation operator.
    /// </summary>
    /// <typeparam name="TType">Operand type.</typeparam>
    /// <typeparam name="TGrad">Type used for gradient calculations.</typeparam>
    public struct NegOp<TType, TGrad> : IBackwardOne<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        /// <summary>
        /// Negates the operand.
        /// </summary>
        /// <param name="left">Operand.</param>
        public static TType ApplyCpu(TType left) => -left;

        /// <summary>
        /// Negates the operand on the <see cref="Accelerator"/>
        /// </summary>
        /// <param name="idx"><see cref="Accelerator"/> index.</param>
        /// <param name="left">Operand.</param>
        /// <param name="output">Result of the negation.</param>
        public static void ApplyGpu(Index1D idx, ArrayView<TType> left, ArrayView<TType> output) => output[idx] = ApplyCpu(left[idx]);

        /// <summary>
        /// Calculates the gradients of the negation operator.
        /// </summary>
        /// <param name="grad">Gradients to backpropagate.</param>
        /// <param name="left">Operands.</param>
        /// <returns>Gradients of the negations operator.</returns>
        public static TGrad BackwardCpu(TGrad grad, TType left) => -grad;

        /// <summary>
        /// Calculates the gradients of the negation operator on the <see cref="Accelerator"/>.
        /// </summary>
        /// <param name="idx"><see cref="Accelerator"/> index.</param>
        /// <param name="grad">Gradients to backpropagate.</param>
        /// <param name="left">Operands.</param>
        /// <param name="leftGrad">Gradients of the negations operator.</param>
        public static void BackwardGpu(Index1D idx, ArrayView<TGrad> grad, ArrayView<TType> left, ArrayView<TGrad> leftGrad) => leftGrad[idx] += BackwardCpu(grad[idx], left[idx]);
    }
}
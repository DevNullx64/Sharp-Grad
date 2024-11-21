using System.Numerics;

namespace SharpGrad.Operators
{
    /// <summary>
    /// Interface for an aggregator operation.
    /// </summary>
    /// <typeparam name="TOperand1">Type of the first operand.</typeparam>
    /// <typeparam name="TResult">Type of the result.</typeparam>
    /// <remarks>Aggregator operations are unary operations that reduce the dimensionality of the input.</remarks>
    public interface IAgregator<TOperand1, TResult> : IExec
        where TOperand1 : unmanaged, INumber<TOperand1>
        where TResult : unmanaged, INumber<TResult>
    {
        /// <summary>
        /// The resulting <see cref="Shape"/> of the operation.
        /// </summary>
        /// <param name="right">The <see cref="Shape"/> of the first operand. </param>
        /// <returns>The resulting <see cref="Shape"/>. </returns>
        /// <remarks>Operand is not broadcasted.</remarks>
        abstract static Shape ResultingShape(Shape right);

        /// <summary>
        /// Execute the operation.
        /// </summary>
        /// <param name="right">The first operand.</param>
        /// <returns>The result of the operation.</returns>
        abstract static TResult Exec(TOperand1[] right);

        /// <summary>
        /// Compute the gradient of the operation.
        /// </summary>
        /// <param name="right">The first operand.</param>
        /// <param name="grad">The internal gradient.</param>
        /// <returns>The gradients to backpropagate.</returns>
        abstract static TOperand1[] Backward(TOperand1[] right, TResult grad);
    }

}
namespace SharpGrad.Tensors.Operators
{
    /// <summary>
    /// Interface for an operation that takes two operands.
    /// </summary>
    /// <typeparam name="TOperand1">Type of the first operand.</typeparam>
    /// <typeparam name="TOperand2">Type of the second operand.</typeparam>
    /// <typeparam name="TResult">Type of the result.</typeparam>
    public interface IExecutor2<TOperand1, TOperand2, TResult> : IExecutor
    {
        /// <summary>
        /// The resulting <see cref="Shape"/> of the operation.
        /// </summary>
        /// <param name="operand1">The <see cref="Shape"/> of the first operand. </param>
        /// <param name="operand2">The <see cref="Shape"/> of the second operand. </param>
        /// <returns>The resulting <see cref="Shape"/>. </returns>
        /// <remarks>Operand is not broadcasted.</remarks>
        abstract static Shape ResultingShape(Shape operand1, Shape operand2);

        /// <summary>
        /// Execute the operation.
        /// </summary>
        /// <param name="operand1">The first operand.</param>
        /// <param name="operand2">The second operand.</param>
        /// <returns>The result of the operation.</returns>
        abstract static TResult Exec(TOperand1 operand1, TOperand2 operand2);

        /// <summary>
        /// Compute the gradient of the operation.
        /// </summary>
        /// <param name="operand1">The first operand.</param>
        /// <param name="operand2">The second operand.</param>
        /// <param name="grad">The internal gradient.</param>
        /// <returns>The gradients to backpropagate.</returns>
        abstract static (TOperand1 Operand1, TOperand2 Operand2) Backward(TOperand1 operand1, TOperand2 operand2, TResult grad);
    }
}
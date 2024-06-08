namespace SharpGrad.Tensors.Operators
{
    /// <summary>
    /// Interface for an operation that takes one operand.
    /// </summary>
    /// <typeparam name="TOperand">The type of the operand.</typeparam>
    /// <typeparam name="TResult">The type of the result.</typeparam>
    public interface IExecutor1<TOperand, TResult>
    {
        /// <summary>
        /// Compute the resulting <see cref="Shape"/> of the operation.
        /// </summary>
        /// <param name="operand">The <see cref="Shape"/> of the operand. </param>
        /// <returns></returns>
        abstract static Shape ResultingShape(Shape operand);

        /// <summary>
        /// Execute the operation.
        /// </summary>
        /// <param name="operand"></param>
        /// <returns></returns>
        abstract static TResult Exec(TOperand operand);

        /// <summary>
        /// Compute the gradient of the operation.
        /// </summary>
        /// <param name="operand">The operand.</param>
        /// <param name="grad">The internal gradient.</param>
        /// <returns>The gradient to backpropagate.</returns>
        abstract static TOperand Backward(TOperand operand, TResult grad);
    }

}
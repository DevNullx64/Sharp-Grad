namespace SharpGrad.Tensors.Operators
{
    /// <summary>
    /// Interface for an operation that takes two operands.
    /// </summary>
    /// <typeparam name="TOperand1">Type of the first operand.</typeparam>
    /// <typeparam name="TOperand2">Type of the second operand.</typeparam>
    /// <typeparam name="TResult">Type of the result.</typeparam>
    public interface IExecOperation<TOperand1, TOperand2, TResult> : IExec
        where TOperand1 : struct
        where TOperand2 : struct
    {
        /// <summary>
        /// The resulting <see cref="Shape"/> of the operation.
        /// </summary>
        /// <param name="left">The <see cref="Shape"/> of the first operand. </param>
        /// <param name="right">The <see cref="Shape"/> of the second operand. </param>
        /// <returns>The resulting <see cref="Shape"/>. </returns>
        /// <remarks>Operand is not broadcasted.</remarks>
        abstract static Shape ResultingShape(Shape left, Shape right);

        /// <summary>
        /// Execute the operation.
        /// </summary>
        /// <param name="left">The first operand.</param>
        /// <param name="right">The second operand.</param>
        /// <returns>The result of the operation.</returns>
        abstract static TResult Exec(TOperand1 left, TOperand2 right);

        abstract static BackwardNeedOperand BackwardOperand { get; }

        /// <summary>
        /// Compute the gradient of the operation for both operands.
        /// </summary>
        /// <param name="left">Value of the first operand.</param>
        /// <param name="right">Value of the second operand.</param>
        /// <param name="grad">Gradient of the operation.</param>
        /// <returns>The gradient of the operation for, respectively, the first and second operands.</returns>
        abstract static (TResult, TResult) Backward(TOperand1 left, TOperand2 right, TResult grad);
    }
}
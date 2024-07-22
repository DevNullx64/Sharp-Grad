namespace SharpGrad.Tensors.Operators
{
    /// <summary>
    /// Enum for the need of the backward operation for the operands.
    /// </summary>
    public enum BackwardNeedOperand
    {
        /// <summary>
        /// No need for the operand.
        /// </summary>
        None,
        /// <summary>
        /// Need the left operand.
        /// </summary>
        Left,
        /// <summary>
        /// Need the right operand.
        /// </summary>
        Right,
        /// <summary>
        /// Need both operands.
        /// </summary>
        Both
    }
}
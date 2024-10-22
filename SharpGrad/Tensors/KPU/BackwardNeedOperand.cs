namespace SharpGrad.Tensors.KPU
{
    /// <summary>
    /// Enum for the need of the backward operation for the tensors.
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
        /// Need both tensors.
        /// </summary>
        Both
    }
}
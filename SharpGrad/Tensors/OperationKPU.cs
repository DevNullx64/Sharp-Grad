namespace SharpGrad.Tensors
{
    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="indexOperand1">Left operand</param>
    /// <param name="indexOperand2">Output of the operation</param>
    public readonly struct OperationKPU(OpCode opCode, KPUIndex result, KPUIndex indexOperand1, KPUIndex indexOperand2)
    {
        public OperationKPU(OpCode opCode, KPUIndex result, KPUIndex indexOperand1)
            : this(opCode, result, indexOperand1, KPUIndex.Empty)
        { }

        /// <inheritdoc/>
        public OpCode OpCode => opCode;

        /// <summary>
        /// Index of the result
        /// </summary>
        /// <remarks>Negative values are used as indices of the register</remarks>
        public KPUIndex IndexResult => result;

        /// <summary>
        /// Index of the first operand
        /// </summary>
        /// <remarks>Negative values are used as indices of the register</remarks>
        public KPUIndex IndexOperand1 => indexOperand1;

        /// <summary>
        /// Index of the second operand
        /// </summary>
        /// <remarks>Negative values are used as indices of the register</remarks>
        public KPUIndex IndexOperand2 => indexOperand2;
    }

}
using System;

namespace SharpGrad.Tensors.KPU
{
    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="result">Index of the result</param>
    /// <param name="leftOperand">Left operand</param>
    /// <param name="rightOperand">Right operand</param>
    public readonly struct OperationKPU(OpCode opCode, ResultIndex result, OperandIndex leftOperand, OperandIndex rightOperand)
    {
        public OperationKPU(OpCode opCode, ResultIndex result, OperandIndex indexOperand1)
            : this(opCode, result, indexOperand1, OperandIndex.Empty)
        { }

        /// <inheritdoc/>
        public readonly OpCode OpCode = opCode;

        /// <summary>
        /// Index of the result
        /// </summary>
        /// <remarks>Sources must be <see cref="OperandIndexSource.Cache"/> or <see cref="OperandIndexSource.Output"/></remarks>
        public readonly ResultIndex IndexResult = result;

        /// <summary>
        /// Index of the first operand
        /// </summary>
        public readonly OperandIndex LeftOperand = leftOperand;

        /// <summary>
        /// Index of the second operand
        /// </summary>
        public readonly OperandIndex RightOperand = rightOperand;
    }

}
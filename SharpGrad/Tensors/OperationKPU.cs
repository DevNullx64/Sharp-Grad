using System.Collections.Generic;
using System.Data;
using System.Reflection.Emit;
using System.Runtime.InteropServices.JavaScript;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="indexOperand1">Left operand</param>
    /// <param name="indexOperand2">Result of the operation</param>
    public readonly struct OperationKPU(OpCode opCode, short result, short indexOperand1, short indexOperand2 = OperationKPU.NoOperand)
    {
        /// <summary>
        /// Value used to represent an empty index
        /// </summary>
        public const short NoOperand = short.MinValue;

        /// <inheritdoc/>
        public OpCode OpCode => opCode;

        /// <summary>
        /// Index of the result
        /// </summary>
        /// <remarks>Negative values are used as indices of the register</remarks>
        public short IndexResult => result;

        /// <summary>
        /// Index of the first operand
        /// </summary>
        /// <remarks>Negative values are used as indices of the register</remarks>
        public short IndexOperand1 => indexOperand1;

        /// <summary>
        /// Index of the second operand
        /// </summary>
        /// <remarks>Negative values are used as indices of the register</remarks>
        public short IndexOperand2 => indexOperand2;
    }
}
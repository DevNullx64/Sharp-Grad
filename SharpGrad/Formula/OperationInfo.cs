using SharpGrad.Operators;
using System.Numerics;

namespace SharpGrad.Formula
{
    public readonly struct OperationInfo<TResult>(
        OpCode opCode,
        byte shapeIndex,
        BIndex<ushort> outputIndex,
        MultiIndex<SourceOfOperand> leftIndex,
        MultiIndex<SourceOfOperand> rightIndex,
        BIndex<ushort> gradientIndex)
        where TResult : unmanaged, INumber<TResult>
    {
        /// <summary>
        /// The operation code.
        /// </summary>
        public readonly OpCode OpCode = opCode;

        public readonly byte ShapeIndex = shapeIndex;

        /// <summary>
        /// Index where store the output. -1 if the output is not needed.
        /// </summary>
        public readonly BIndex<ushort> OutputIndex = outputIndex;

        /// <summary>
        /// The index of the leftIndex operandIndex.
        /// </summary>
        public readonly MultiIndex<SourceOfOperand> LeftIndex = leftIndex;

        /// <summary>
        /// The index of the rightIndex operandIndex.
        /// </summary>
        public readonly MultiIndex<SourceOfOperand> RightIndex = rightIndex;

        public readonly BIndex<ushort> GradientIndex = gradientIndex;
    }
}
using SharpGrad.Tensors.KPU;
using System.Numerics;

namespace SharpGrad.Tensors.Formula
{
    public readonly struct OperationInfo<TResult>(
        OpCode opCode,
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
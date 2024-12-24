using SharpGrad.Operators;
using System.Numerics;

namespace SharpGrad.Formula.Internal
{
    public readonly struct InternalOperation<TResult>(
        OpCode opCode,
        byte shapeIndex,
        BIndex<byte> outputIndex,
        OperandIndex<sbyte> leftIndex,
        OperandIndex<sbyte> rightIndex,
        BIndex<byte> gradientIndex)
        where TResult : unmanaged, INumber<TResult>
    {
        /// <summary>
        /// The operation code.
        /// </summary>
        public readonly OpCode OpCode = opCode;

        public readonly bool IsFunction => OpCode.HasFlag(OpCode.IsUnary);
        public readonly bool IsOperator => OpCode.HasFlag(OpCode.IsBinary);
        public readonly bool IsReduction => OpCode.HasFlag(OpCode.IsReduction);


        public readonly BIndex<byte> ShapeIdx = shapeIndex;
        public readonly bool IsScalar => ShapeIdx.IsEmpty;

        /// <summary>
        /// Index where store the output. -1 if the output is not needed.
        /// </summary>
        public readonly BIndex<byte> OutputIdx = outputIndex;
        public readonly bool IsOutput => !OutputIdx.IsEmpty;

        /// <summary>
        /// The index of the left operand.
        /// </summary>
        public readonly OperandIndex<sbyte> LeftIdx = leftIndex;

        /// <summary>
        /// The index of the right operand.
        /// </summary>
        public readonly OperandIndex<sbyte> RightIdx = rightIndex;
        public readonly bool HasRight => !RightIdx.IsEmpty;

        public readonly BIndex<byte> GradientIndex = gradientIndex;
        public readonly bool HasGradient => !GradientIndex.IsEmpty;
    }
}
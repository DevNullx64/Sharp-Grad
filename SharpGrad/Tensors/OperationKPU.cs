using System;

namespace SharpGrad.Tensors
{
    public readonly struct OperationKPUIndex
    {
        public static readonly OperationKPUIndex Empty = new(-1, SourceList.None);
        public static readonly short MinValue = 0;
        private const ushort Shift = 14;
        public static readonly short MaxValue = (1 << Shift) - 1;

        public enum SourceList : short
        {
            None = -1,
            Operand = 0,
            Operation = 1,
            Cache = 2,
            Result = 3
        }

        private readonly ushort value;

        public readonly bool IsEmpty => (short)value == -1;
        public readonly short Value => (short)(IsEmpty ? -1 : value & MaxValue);
        public readonly SourceList Source => IsEmpty ? SourceList.None : (SourceList)(value >> Shift);

        public OperationKPUIndex(short value, SourceList source)
        {
            if(value > MaxValue)
                throw new ArgumentOutOfRangeException(nameof(value), $"Value must be less than {MaxValue}.");
            this.value = (source == SourceList.None || value < 0)
                ? ushort.MaxValue
                : (ushort)((ushort)source << Shift | (ushort)value);
        }
    }

    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="indexOperand1">Left operand</param>
    /// <param name="indexOperand2">Result of the operation</param>
    public readonly struct OperationKPU(OpCode opCode, short result, short indexOperand1, short indexOperand2)
    {
        /// <summary>
        /// Value used to represent an empty index
        /// </summary>
        public const short NoOperand = short.MinValue;

        public OperationKPU(OpCode opCode, short result, short indexOperand1)
            : this(opCode, result, indexOperand1, NoOperand)
        { }

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
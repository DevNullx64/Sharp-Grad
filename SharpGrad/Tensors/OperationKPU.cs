using ILGPU;
using ILGPU.Runtime;
using System;
using System.Data;
using System.Numerics;
using System.Reflection.Emit;
using System.Runtime.InteropServices.JavaScript;

namespace SharpGrad.Tensors
{
    public enum NativeType : int
    {
        Unknown = 0,
        Bits8 = 1, Bits16 = 2, Bits32 = 3, Bits64 = 4,
        Integer = 8, UInteger = 16, FloatingPoint = 24,

        Double = Bits64 | FloatingPoint,
        Float = Bits32 | FloatingPoint,
        Long = Bits64 | Integer,
        ULong = Bits64 | UInteger,
        Int = Bits32 | Integer,
        UInt = Bits32 | UInteger,
        Short = Bits16 | Integer,
        UShort = Bits16 | UInteger,
        Byte = Bits16 | Integer,
        SByte = Bits16 | Integer,
    }

    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    public enum OpCode : int
    {
        Store,
        Reset,
        Add,
        Sub,
        Neg,
        Mul,
        Div,
        Pow,
        Log,
        Exp,
    }

    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="left">Left operand</param>
    /// <param name="right">Right operand</param>
    /// <param name="result">Result of the operation</param>
    /// <remarks>Operations are performed in-place</remarks>
    public interface IOpeartion<T>
    {
        /// <summary>
        /// Operation to perform
        /// </summary>
        OpCode OpCode { get; }
        /// <summary>
        /// Left operand
        /// </summary>
        T Left { get; }
        /// <summary>
        /// Right operand
        /// </summary>
        T Right { get; }
        /// <summary>
        /// Result of the operation
        /// </summary>
        T Result { get; }
    }

    /// <summary>
    /// An operation to perform
    /// </summary>
    /// <typeparam name="TView">The type of the view</typeparam>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="left">Left operand</param>
    /// <param name="right">Right operand</param>
    /// <param name="result">Result of the operation</param>
    /// <remarks>Operations are performed in-place</remarks>
    public readonly struct Operation<TView>(OpCode opCode, TView left, TView right, TView result) : IOpeartion<TView>
        where TView: struct, IArrayView
    {
        /// <inheritdoc/>
        public OpCode OpCode => opCode;

        private readonly TView left = left.IsValid && left.Length == right.Length ? left : throw new ArgumentException($"Expected {nameof(left)} and {nameof(right)} to have the same length, got {left.Length} and {right.Length}");
        /// <inheritdoc/>
        public TView Left => left;

        private readonly TView right = right.IsValid && right.Length == result.Length ? right : throw new ArgumentException($"Expected {nameof(right)} and {nameof(result)} to have the same length, got {right.Length} and {result.Length}");
        /// <inheritdoc/>
        public TView Right => right;

        /// <inheritdoc/>
        public TView Result => result;
    }

    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="left">Left operand</param>
    /// <param name="right">Right operand</param>
    /// <param name="result">Result of the operation</param>
    /// <remarks>Operations are performed in-place</remarks>
    public readonly struct OperationKPU(OpCode opCode, int left, int right, int result): IOpeartion<int>
    {
        /// <inheritdoc/>
        public OpCode OpCode => opCode;

        /// <inheritdoc/>
        /// <remarks>Set to -1 to use the accumulator</remarks>
        public int Left => left;

        /// <inheritdoc/>
        /// <remarks>Set to -1 to use the accumulator</remarks>
        public int Right => right;

        /// <inheritdoc/>
        /// <remarks>Set to -1 to use the accumulator</remarks>
        public int Result => result;
    }
}
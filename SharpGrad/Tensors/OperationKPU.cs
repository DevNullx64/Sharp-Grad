using ILGPU;
using ILGPU.Runtime;
using System;
using System.Data;
using System.Numerics;
using System.Reflection.Emit;
using System.Runtime.InteropServices.JavaScript;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    public enum OpCode : int
    {
        Load,
        Save,
        Add,
        Sub,
        Mul,
        Div,
    }

    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="left">Left operand</param>
    /// <param name="right">Right operand</param>
    /// <param name="result">Result of the operation</param>
    /// <remarks>Operations are performed in-place</remarks>
    public interface IOpeartion<TType>
    {
        /// <summary>
        /// Operation to perform
        /// </summary>
        OpCode OpCode { get; }
        /// <summary>
        /// Left operand
        /// </summary>
        TType Left { get; }
        /// <summary>
        /// Right operand
        /// </summary>
        TType Right { get; }
        /// <summary>
        /// Result of the operation
        /// </summary>
        TType Result { get; }
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
}
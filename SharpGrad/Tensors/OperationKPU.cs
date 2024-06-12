using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
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
    public enum OpCode : short
    {
        /// <summary>
        /// If set, the operation is commutative. Otherwise, it is not.
        /// </summary>
        Commutative = 0x80,

        /// <summary>
        /// If set, the operation is unary. Otherwise, it is binary.
        /// </summary>
        Unary = 0x40,

        Reset = 0 | Unary,
        Store = 1 | Unary,
        Add = 2 | Commutative,
        Sub = 3,
        Neg = Sub | Unary,
        Mul = 4 | Commutative,
        Div = 5,
        Pow = 8,
        Log = 9 | Unary,
        Exp = 10 | Unary,
        Sum = 11 | Unary,
        Prod = 12 | Unary,
    }

    public interface IBufferOperande
    {
        long Length { get; }
    }

    public readonly struct MemBufferOperande<T>(MemoryBuffer1D<T, Stride1D.Dense> buffer) : IBufferOperande
        where T : unmanaged, INumber<T>
    {
        public long Length => buffer.Length;
        public MemoryBuffer1D<T, Stride1D.Dense> Buffer => buffer;
    }

    public interface IBufferOperation : IBufferOperande
    {
        /// <summary>
        /// Operation to perform
        /// </summary>
        OpCode OpCode { get; }
        IBufferOperande Buffer1 { get; }
        IBufferOperande Buffer2 { get; }
    }

    /// <summary>
    /// An operation to perform using the KPU
    /// </summary>
    /// <param name="opCode">Operation to perform</param>
    /// <param name="index1">Left operand</param>
    /// <param name="index2">Result of the operation</param>
    /// <remarks>Operations are performed in-place</remarks>
    public readonly struct OperationKPU(OpCode opCode, short index1, short index2)
    {
        /// <inheritdoc/>
        public OpCode OpCode => opCode;

        /// <inheritdoc/>
        /// <remarks>Negative values are used as indices of the accumulator</remarks>
        public short Index1 => index1;

        /// <inheritdoc/>
        /// <remarks>Negative values are used as indices of the accumulator</remarks>
        public short Index2 => index2;

        public readonly short MinIndex => Math.Min(index1, index2);
        public readonly short MaxIndex => Math.Max(index1, index2);
    }
}
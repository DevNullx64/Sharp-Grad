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
        Neg = 4 | Unary,
        Mul = 5 | Commutative,
        Div = 6,
        Pow = 8,
        Log = 9 | Unary,
        Exp = 10 | Unary,
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
    public abstract class BufferOperationBase(OpCode opCode, long length, object buffer1, object buffer2) : IBufferOperation
        {
        public OpCode OpCode { get; } = opCode;
        public long Length { get; } = length;
        public object Buffer1 { get; } = buffer1;
        public object Buffer2 { get; } = buffer2;

        public override bool Equals(object? obj)
        {
            if (ReferenceEquals(this, obj))
                return true;

            if (obj is BufferOperationBase other)
            {
                if(OpCode == other.OpCode && Length == other.Length)
                {
                    if (Buffer1.Equals(other.Buffer1) && Buffer2.Equals(other.Buffer2))
                        return true;
                    if (OpCode.HasFlag(OpCode.Commutative))
                        return Buffer1.Equals(other.Buffer2) && Buffer2.Equals(other.Buffer1);
                }
            }
            return false;
        }

        public override int GetHashCode()
        {
            int result = HashCode.Combine(OpCode, Length) ^ Buffer1.GetHashCode();
            if (OpCode.HasFlag(OpCode.Commutative))
                return result ^ Buffer2.GetHashCode();
            else
                return result ^ (Buffer2.GetHashCode() * 31);
        }
    }

    public class BufferOperationMemMem<T>(OpCode opCode, MemoryBuffer1D<T, Stride1D.Dense> buffer1, MemoryBuffer1D<T, Stride1D.Dense> buffer2):
        BufferOperationBase(opCode, Math.Max(buffer1.Length, buffer2.Length), buffer1, buffer2),
        IBufferOperation<T>
        where T : unmanaged, INumber<T>
    {
        public new MemoryBuffer1D<T, Stride1D.Dense> Buffer1 => buffer1.Length == 1 || buffer1.Length == buffer2.Length
            ? buffer1
            : throw new ArgumentException($"{nameof(buffer1)}({buffer1.Length}) must be the same length as {nameof(buffer2)}({buffer2.Length})");
        public new MemoryBuffer1D<T, Stride1D.Dense> Buffer2 => buffer2.Length == 1 || buffer2.Length == buffer1.Length
            ? buffer2
            : throw new ArgumentException($"{nameof(buffer2)}({buffer2.Length}) must be the same length as {nameof(buffer1)}({buffer1.Length})";

        IBufferOperation IBufferOperation<T>.Buffer1 => this.Buffer1;

        IBufferOperation IBufferOperation<T>.Buffer2 => throw new NotImplementedException();
    }

    public class BufferOperationMemAcc<T>(MemoryBuffer1D<T, Stride1D.Dense> buffer1, BufferOperationMemMem<T> buffer2) : IBufferOperation
        where T : unmanaged, INumber<T>
    {
        public long Length { get; } = Math.Max(buffer1.Length, buffer2.Length);
        public readonly MemoryBuffer1D<T, Stride1D.Dense> Buffer1 => buffer1.Length == 1 || buffer1.Length == buffer2.Length
            ? buffer1
            : throw new ArgumentException($"{nameof(buffer1)}({buffer1.Length}) must be the same length as {nameof(buffer2)}({buffer2.Length})");
        public readonly BufferOperationMemMem<T> Buffer2 => buffer2.Length == 1 || buffer2.Length == buffer1.Length
            ? buffer2
            : throw new ArgumentException($"{nameof(buffer2)}({buffer2.Length}) must be the same length as {nameof(buffer1)}({buffer1.Length})");

        readonly object IBufferOperation.Buffer1 => Buffer1;
        readonly object IBufferOperation.Buffer2 => Buffer2;
    }

    public class BufferOperationAccMem<T>(BufferOperationMemMem<T> buffer1, MemoryBuffer1D<T, Stride1D.Dense> buffer2) : IBufferOperation
        where T : unmanaged, INumber<T>
    {
        public long Length { get; } = Math.Max(buffer1.Length, buffer2.Length);
        public readonly BufferOperationMemMem<T> Buffer1 => buffer1.Length == 1 || buffer1.Length == buffer2.Length
            ? buffer1
            : throw new ArgumentException($"{nameof(buffer1)}({buffer1.Length}) must be the same length as {nameof(buffer2)}({buffer2.Length})");
        public readonly MemoryBuffer1D<T, Stride1D.Dense> Buffer2 => buffer2.Length == 1 || buffer2.Length == buffer1.Length
            ? buffer2
            : throw new ArgumentException($"{nameof(buffer2)}({buffer2.Length}) must be the same length as {nameof(buffer1)}({buffer1.Length})");

        readonly object IBufferOperation.Buffer1 => Buffer1;
        readonly object IBufferOperation.Buffer2 => Buffer2;
    }

    public class BufferOperationAccAcc<T>(BufferOperationMemMem<T> buffer1, BufferOperationMemMem<T> buffer2) : IBufferOperation
        where T : unmanaged, INumber<T>
    {
        public long Length { get; } = Math.Max(buffer1.Length, buffer2.Length);
        public readonly BufferOperationMemMem<T> Buffer1 => buffer1.Length == 1 || buffer1.Length == buffer2.Length
            ? buffer1
            : throw new ArgumentException($"{nameof(buffer1)}({buffer1.Length}) must be the same length as {nameof(buffer2)}({buffer2.Length})");
        public readonly BufferOperationMemMem<T> Buffer2 => buffer2.Length == 1 || buffer2.Length == buffer1.Length
            ? buffer2
            : throw new ArgumentException($"{nameof(buffer2)}({buffer2.Length}) must be the same length as {nameof(buffer1)}({buffer1.Length})");

        readonly object IBufferOperation.Buffer1 => Buffer1;
        readonly object IBufferOperation.Buffer2 => Buffer2;
    }


    public class BufferScript<T>
        where T : unmanaged, INumber<T>
    {
        protected List<IBufferOperation<T>> operation;
        public IReadOnlyList<IBufferOperation<T>> Operations => operation;

        public OperationKPU ForKPU()
        { 
            Dictionary<IBufferOperation<T>, int> operationUsage = new();
            Dictionary<MemoryBuffer1D<T, Stride1D.Dense>, int> memoryBuffer1Ds = [];
            for(int i = 0; i < Operations.Count; i++)
            {
                if(operationUsage.TryGetValue(Operations[i], out int count))
                    operationUsage[Operations[i]] = count + 1;
                else
                    operationUsage[Operations[i]] = 1;

                var op = Operations[i];
                if (op is BufferOperationMemMem<T> memMem)
                {
                    if(memoryBuffer1Ds.TryGetValue(memMem.Buffer1, out int c1))
                        memoryBuffer1Ds[memMem.Buffer1] = c1 + 1;
                    else
                        memoryBuffer1Ds[memMem.Buffer1] = 1;

                    if(memoryBuffer1Ds.TryGetValue(memMem.Buffer2, out int c2))
                        memoryBuffer1Ds[memMem.Buffer2] = c2 + 1;
                    else
                        memoryBuffer1Ds[memMem.Buffer2] = 1;
                }
                else if (op is BufferOperationMemAcc<T> memAcc)
                {
                    if (memoryBuffer1Ds.TryGetValue(memAcc.Buffer1, out int c1))
                        memoryBuffer1Ds[memAcc.Buffer1] = c1 + 1;
                    else
                        memoryBuffer1Ds[memAcc.Buffer1] = 1;
                }
                else if (op is BufferOperationAccMem<T> accMem)
                {
                    if (memoryBuffer1Ds.TryGetValue(accMem.Buffer2, out int c2))
                        memoryBuffer1Ds[accMem.Buffer2] = c2 + 1;
                    else
                        memoryBuffer1Ds[accMem.Buffer2] = 1;
                }
            }


        }
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
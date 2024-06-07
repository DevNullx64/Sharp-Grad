using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SharpGrad.Tensors
{
    public enum CastCombination: int
    {
        // From
        FromDouble = 1,
        FromFloat = 2,
        FromLong = 3,
        FromULong = 4,
        FromInt = 5,
        FromUInt = 6,
        FromShort = 7,
        FromUShort = 8,
        FromByte = 9,
        FromSByte = 10,
        // To
        ToDouble = FromDouble << 4,
        ToFloat = FromFloat << 4,
        ToLong = FromLong << 4,
        ToULong = FromULong << 4,
        ToInt = FromInt << 4,
        ToUInt = FromUInt << 4,
        ToShort = FromShort << 4,
        ToUShort = FromUShort << 4,
        ToByte = FromByte << 4,
        ToSByte = FromSByte << 4,
    }

    public static class Acc
    {
        private static Context GetContext()
        {
            Context result = Context.Create(builder => builder.AllAccelerators());
            Debug.WriteLine($"Context created: {result}");
            return result;
        }
        private static readonly Context context = GetContext();

        private static Device GetDevice(Context context)
        {
            Device result = context.GetPreferredDevice(preferCPU: false);
            Debug.WriteLine($"Device created: {result}");
            return result;
        }
        private static readonly Device device = GetDevice(context);
        private static readonly Accelerator Accelerator = device.CreateAccelerator(context);
        public static void Synchronize() => Accelerator.Synchronize();

        public static void PrintInformation(TextWriter writer) { Accelerator.PrintInformation(writer); }


        #region Exec
        //public static void Exec<T, U>(
        //    Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<U, Stride1D.Dense>> func,
        //    ArrayView1D<T, Stride1D.Dense> left,
        //    ArrayView1D<U, Stride1D.Dense> result)
        //    where T : unmanaged, INumber<T>
        //    where U : unmanaged, INumber<U>
        //{
        //    Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<U, Stride1D.Dense>> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel(func);
        //    loadedKernel(left.IntExtent, left, result);
        //    Accelerator.Synchronize();
        //}
        public static void Exec<T>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> func,
            ArrayView1D<T, Stride1D.Dense> left,
            ArrayView1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left, result);
            Accelerator.Synchronize();
        }
        public static void Exec<T>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> func,
            ArrayView1D<T, Stride1D.Dense> left,
            ArrayView1D<T, Stride1D.Dense> right,
            ArrayView1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left, right, result);
            Accelerator.Synchronize();
        }

        public static void Exec<T>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, T, ArrayView1D<T, Stride1D.Dense>> func,
            ArrayView1D<T, Stride1D.Dense> left,
            T right,
            ArrayView1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, T, ArrayView1D<T, Stride1D.Dense>> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left, right, result);
            Accelerator.Synchronize();
        }

        public static void Exec<T>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left,
            MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
            => Exec(func, left.View, result.View);


        private static void ExecKernel<TExec, TOperand1, TResult>(
            Index1D idx,
            ArrayView1D<TOperand1, Stride1D.Dense> operand1,
            ArrayView1D<TResult, Stride1D.Dense> result)
            where TExec : IExecutor1<TOperand1, TResult>
            where TOperand1 : unmanaged, INumber<TOperand1>
            where TResult : unmanaged, INumber<TResult>
        { result[idx] = TExec.Exec(operand1[idx]); }

        private static void ExecKernel<TExec, TOperand1, TOperand2, TResult>(
            Index1D idx,
            ArrayView1D<TOperand1, Stride1D.Dense> operand1,
            ArrayView1D<TOperand2, Stride1D.Dense> operand2,
            ArrayView1D<TResult, Stride1D.Dense> result)
            where TExec : IExecutor2<TOperand1, TOperand2, TResult>
            where TOperand1 : unmanaged, INumber<TOperand1>
            where TOperand2 : unmanaged, INumber<TOperand2>
            where TResult : unmanaged, INumber<TResult>
        { result[idx] = TExec.Exec(operand1[idx], operand2[idx]); }


        public static MemoryBuffer1D<TResult, Stride1D.Dense> Exec<TExec, TOperand1, TResult>(
            MemoryBuffer1D<TOperand1, Stride1D.Dense> operand1)
            where TExec : IExecutor1<TOperand1, TResult>
            where TOperand1 : unmanaged, INumber<TOperand1>
            where TResult : unmanaged, INumber<TResult>
        {
            MemoryBuffer1D<TResult, Stride1D.Dense> result = Allocate1D<TResult>(operand1.Length);
            Action<Index1D, ArrayView1D<TOperand1, Stride1D.Dense>, ArrayView1D<TResult, Stride1D.Dense>> loadedKernel
                = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<TOperand1, Stride1D.Dense>,ArrayView1D<TResult, Stride1D.Dense>>(ExecKernel<TExec, TOperand1, TResult>);
            loadedKernel(result.IntExtent, operand1.View, result.View);
            return result;
        }

        public static MemoryBuffer1D<TResult, Stride1D.Dense> Exec<TExec, TOperand1, TOperand2, TResult>(
            MemoryBuffer1D<TOperand1, Stride1D.Dense> operand1,
            MemoryBuffer1D<TOperand2, Stride1D.Dense> operand2)
            where TExec : IExecutor2<TOperand1, TOperand2, TResult>
            where TOperand1 : unmanaged, INumber<TOperand1>
            where TOperand2 : unmanaged, INumber<TOperand2>
            where TResult : unmanaged, INumber<TResult>
        {
            if (operand1.Length != operand2.Length)
                throw new ArgumentException($"Expected {nameof(operand1)} and {nameof(operand2)} to have the same length, got {operand1.Length} and {operand2.Length}");
            MemoryBuffer1D<TResult, Stride1D.Dense> result = Allocate1D<TResult>(operand1.Length);
            Action<Index1D, ArrayView1D<TOperand1, Stride1D.Dense>, ArrayView1D<TOperand2, Stride1D.Dense>, ArrayView1D<TResult, Stride1D.Dense>> loadedKernel
                = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<TOperand1, Stride1D.Dense>, ArrayView1D<TOperand2, Stride1D.Dense>, ArrayView1D<TResult, Stride1D.Dense>>(ExecKernel<TExec, TOperand1, TOperand2, TResult>);
            loadedKernel(result.IntExtent, operand1.View, operand2.View, result.View);
            return result;
        }


        public static MemoryBuffer1D<T, Stride1D.Dense> Exec<TExec, T>(
            MemoryBuffer1D<T, Stride1D.Dense> operand1, MemoryBuffer1D<T, Stride1D.Dense> operand2)
            where TExec : IExecutor2<T, T, T>
            where T : unmanaged, INumber<T>
            => Exec<TExec, T, T, T>(operand1, operand2);

        public static MemoryBuffer1D<T, Stride1D.Dense> Exec<TExec, T>(
            MemoryBuffer1D<T, Stride1D.Dense> operand1)
            where TExec : IExecutor1<T, T>
            where T : unmanaged, INumber<T>
            => Exec<TExec, T, T>(operand1);

        public static void Exec<T>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left,
            MemoryBuffer1D<T, Stride1D.Dense> right,
            MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
            => Exec(func, left.View, right.View, result.View);

        public static void Exec<T>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, T, ArrayView1D<T, Stride1D.Dense>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left,
            T right,
            MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
            => Exec(func, left.View, right, result.View);


        public static void Exec<T>(
            OpCode[] operations,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            if (left.Length != right.Length || left.Length != result.Length)
                throw new ArgumentException($"Length mismatch: {nameof(left)}:{left.Length}, {nameof(right)}:{right.Length}, {nameof(result)}:{result.Length}");
            Exec(operations, left, right, result);
        }
        public static void Exec<T>(
            OpCode[] operations,
            MemoryBuffer1D<T, Stride1D.Dense> left, T right, MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
            => Exec(operations, left, right, result);
        #endregion

        #region FillKernel
        private static void FillKernel<T>(Index1D idx, ArrayView1D<T, Stride1D.Dense> view, T value)
            where T : unmanaged, INumber<T> { view[idx] = value; }

        public static void Fill<T>(this ArrayView1D<T, Stride1D.Dense> view, T value)
            where T : unmanaged, INumber<T>
        {
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, T> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, T>(FillKernel);
            loadedKernel(view.IntExtent, view, value);
            Accelerator.Synchronize();
        }

        public static void Fill<T>(this MemoryBuffer1D<T, Stride1D.Dense> mem, T value)
            where T : unmanaged, INumber<T>
            => Fill(mem.View, value);
        #endregion

        #region CastKernel

        #region To double
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(double from) => (float)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(float from) => (double)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ToLong(double from) => (long)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<long, Stride1D.Dense> to) => to[idx] = ToLong(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(long from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<long, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong ToULong(double from) => (ulong)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<ulong, Stride1D.Dense> to) => to[idx] = ToULong(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(ulong from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<ulong, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ToInt(double from) => (int)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<int, Stride1D.Dense> to) => to[idx] = ToInt(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(int from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<int, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint ToUInt(double from) => (uint)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<uint, Stride1D.Dense> to) => to[idx] = ToUInt(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(uint from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<uint, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static short ToShort(double from) => (short)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<short, Stride1D.Dense> to) => to[idx] = ToShort(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(short from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<short, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ushort ToUShort(double from) => (ushort)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<ushort, Stride1D.Dense> to) => to[idx] = ToUShort(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(ushort from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<ushort, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static byte ToByte(double from) => (byte)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<byte, Stride1D.Dense> to) => to[idx] = ToByte(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(byte from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<byte, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static sbyte ToSByte(double from) => (sbyte)from;
        private static void CastKernel(Index1D idx, ArrayView1D<double, Stride1D.Dense> from, ArrayView1D<sbyte, Stride1D.Dense> to) => to[idx] = ToSByte(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double ToDouble(sbyte from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<sbyte, Stride1D.Dense> from, ArrayView1D<double, Stride1D.Dense> to) => to[idx] = ToDouble(from[idx]);
        #endregion

        #region To float
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ToLong(float from) => (long)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<long, Stride1D.Dense> to) => to[idx] = ToLong(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(long from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<long, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong ToULong(float from) => (ulong)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<ulong, Stride1D.Dense> to) => to[idx] = ToULong(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(ulong from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<ulong, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ToInt(float from) => (int)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<int, Stride1D.Dense> to) => to[idx] = ToInt(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(int from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<int, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint ToUInt(float from) => (uint)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<uint, Stride1D.Dense> to) => to[idx] = ToUInt(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(uint from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<uint, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static short ToShort(float from) => (short)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<short, Stride1D.Dense> to) => to[idx] = ToShort(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(short from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<short, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ushort ToUShort(float from) => (ushort)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<ushort, Stride1D.Dense> to) => to[idx] = ToUShort(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(ushort from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<ushort, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static byte ToByte(float from) => (byte)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<byte, Stride1D.Dense> to) => to[idx] = ToByte(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(byte from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<byte, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static sbyte ToSByte(float from) => (sbyte)from;
        private static void CastKernel(Index1D idx, ArrayView1D<float, Stride1D.Dense> from, ArrayView1D<sbyte, Stride1D.Dense> to) => to[idx] = ToSByte(from[idx]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float ToFloat(sbyte from) => from;
        private static void CastKernel(Index1D idx, ArrayView1D<sbyte, Stride1D.Dense> from, ArrayView1D<float, Stride1D.Dense> to) => to[idx] = ToFloat(from[idx]);
        #endregion

        #endregion

        #region Memory Managment
        private static readonly HashSet<IAcceleratorBuffer> AcceleratorBuffers = [];
        private static readonly HashSet<MemoryBuffer> MemoryBuffers = [];
        internal static void Dispose(AcceleratorBuffer acceleratorBuffer)
            => AcceleratorBuffers.Remove(acceleratorBuffer);

        internal static void Dispose<T>(AcceleratorBuffer<T> acceleratorBuffer)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
            => AcceleratorBuffers.Remove(acceleratorBuffer);

        public static MemoryBuffer1D<T, Stride1D.Dense> Allocate1D<T>(long length)
            where T : unmanaged
        {
            bool oom = false;
            try
            {
                Stride1D.Dense stride = default;
                MemoryBuffer1D<T, Stride1D.Dense> buffer = Accelerator.Allocate1D<T, Stride1D.Dense>(length, stride);
                MemoryBuffers.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                oom = true;
            }

            oom = false;
            long count = length;
            foreach (var a in AcceleratorBuffers.Where(e => e.Location == BufferLocation.Accelerator).OrderBy(e => e.LastAccess))
            {
                if (count <= 0)
                    break;
                a.Location = BufferLocation.Ram;
                count -= a.Length;
            }
            return Allocate1D<T>(length);
        }
        public static MemoryBuffer1D<T, Stride1D.Dense> Allocate1D<T>(T[] data)
            where T : unmanaged, INumber<T>
        {
            try
            {
                MemoryBuffer1D<T, Stride1D.Dense> buffer = Allocate1D<T>(data.LongLength);
                buffer.CopyFromCPU(data);
                MemoryBuffers.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }

        public static AcceleratorBuffer<T> GetAcceleratorBuffer<T>(long length)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = AcceleratorBuffer<T>.Create(length);
                AcceleratorBuffers.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }

        public static AcceleratorBuffer<T> GetAcceleratorBuffer<T>(T[] data)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = AcceleratorBuffer<T>.Create(data);
                AcceleratorBuffers.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }
        public static AcceleratorBuffer<T> GetAcceleratorBuffer<T>(AcceleratorBuffer<T> data)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = AcceleratorBuffer<T>.Create(data);
                AcceleratorBuffers.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }

        public static AcceleratorBuffer<T> GetAcceleratorBuffer<T>(MemoryBuffer1D<T, Stride1D.Dense> data)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = AcceleratorBuffer<T>.Create(data);
                AcceleratorBuffers.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }
        #endregion
    }
}

using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using SharpGrad.Tensors.Operators;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public static class KernelProcessUnit
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


        private static List<AcceleratorBuffer> Allocs = [];
        private static MemoryBuffer1D<T, Stride1D.Dense> MemoryBuffer1D<T>(long length)
            where T : unmanaged, INumber<T>
        {
            try
            {
                return Accelerator.Allocate1D<T, Stride1D.Dense>(length, new Stride1D.Dense());
            }
            catch { }
            FreeAcceleratorMemory(length);
            return MemoryBuffer1D<T>(length);
        }
        public static void FreeAcceleratorMemory(long length = 0)
        {
            lock (Allocs)
            {
                if (Allocs.Count == 0)
                    return;
                long toFree = length < 1 ? long.MaxValue : length;
                foreach (var buf in Allocs.Where(e => e.Location == BufferLocation.Accelerator).OrderBy(e => e.LastAccess))
                {
                    buf.Location = BufferLocation.Ram;
                    toFree -= buf.Length;
                    if (toFree <= 0)
                        break;
                }
                Synchronize();
            }
        }


        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="operation">Operation to perform</param>
        /// <param name="left">Left operand</param>
        /// <param name="right">Right operand</param>
        /// <param name="result">Result of the operation</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void KPU<T>(OpCode operation, T left, T right, out T result)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            result = operation switch
            {
                OpCode.Store => left,
                OpCode.Add => AddOp<T>.Exec(left, right),
                OpCode.Sub => SubOp<T>.Exec(left, right),
                OpCode.Mul => MulOp<T>.Exec(left, right),
                OpCode.Div => DivOp<T>.Exec(left, right),
                OpCode.Pow => PowOp<T>.Exec(left, right),
                OpCode.Neg => NegOp<T>.Exec(left),
                OpCode.Log => LogOp<T>.Exec(left),
                OpCode.Exp => ExpOp<T>.Exec(left),
                _ => T.Zero,
            };
        }

        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="idx">GPU Index</param>
        /// <param name="ops">Operations to perform</param>
        /// <param name="tensors">Tensors to operate on</param>
        public static void KPU<T>(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<T, Stride2D.DenseX> tensors, SpecializedValue<int> accumulatorCount)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            T[] accumulator = new T[accumulatorCount];

            for (int i = 0; i < ops.Length; i++)
            {
                OperationKPU op = ops[i];
                T left = op.Left < 0 ? accumulator[-op.Left + 1] : tensors[op.Left, idx];
                T right = op.Right < 0 ? accumulator[-op.Left + 1] : tensors[op.Right, idx];
                KPU(op.OpCode, left, right, out T result);
                if(op.Result < 0)
                    accumulator[-op.Left + 1] = result;
                else
                    tensors[op.Result, idx] = result;
            }
        }
    }
}
using ILGPU.Runtime;
using ILGPU;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using SharpGrad.Tensors.Operators;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit
    {
        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="operation">Operation to perform</param>
        /// <param name="operand">Left operand</param>
        /// <param name="result">Result of the operation</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void KPU<T>(OpCode operation, ref T result, T operand)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            result = operation switch
            {
                OpCode.Reset => T.Zero,
                OpCode.Store => operand,
                OpCode.Add => AddOp<T>.Exec(result, operand),
                OpCode.Sub => SubOp<T>.Exec(result, operand),
                OpCode.Mul => MulOp<T>.Exec(result, operand),
                OpCode.Div => DivOp<T>.Exec(result, operand),
                OpCode.Pow => PowOp<T>.Exec(result, operand),
                OpCode.Neg => NegOp<T>.Exec(operand),
                OpCode.Log => LogOp<T>.Exec(operand),
                OpCode.Exp => ExpOp<T>.Exec(operand),
                _ => T.Zero,
            };
        }

        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="idx">GPU Index</param>
        /// <param name="ops">Operations to perform</param>
        /// <param name="tensors">Tensors to operate on</param>
        /// <param name="registerCount">Number of registers to use</param>
        public static void KPU<T>(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<T, Stride2D.DenseY> tensors, SpecializedValue<short> registerCount)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            T[] register = new T[registerCount];

            for (int i = 0; i < ops.Length; i++)
            {
                OperationKPU op = ops[i];
                if (op.Index1 >= 0)
                {
                    if (op.Index2 >= 0)
                        KPU(op.OpCode, ref tensors[op.Index1, idx], tensors[op.Index2, idx]);
                    else
                        KPU(op.OpCode, ref tensors[op.Index1, idx], register[-op.Index2 - 1]);
                    continue;
                }
                else
                {
                    if (op.Index2 >= 0)
                        KPU(op.OpCode, ref register[-op.Index1 - 1], tensors[op.Index2, idx]);
                    else
                        KPU(op.OpCode, ref register[-op.Index1 - 1], register[-op.Index2 - 1]);
                    continue;
                }
            }
        }

        public MemoryBuffer1D<T, Stride1D.Dense> Exec<T>(OperationKPU[] operations)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var needAccumulator = operations.Min(e => Math.Min(e.Index1, e.Index2));
            if (needAccumulator < 0)
                needAccumulator = (short)-needAccumulator;
            else
                needAccumulator = 0;

            var ops = Accelerator.Allocate1D<OperationKPU, Stride1D.Dense>(operations.Length, new Stride1D.Dense());
            ops.CopyFromCPU(operations);

            var tensors = Accelerator.Allocate2DDenseY<T>(new(operations.Max(e => e.Index1), operations.Length));

            // TODO : Copy input

            KPU(new Index1D(tensors.IntExtent.Y), ops.View, tensors.View, new SpecializedValue<short>(needAccumulator));

            // TODO : Return tensor
            return tensors[0];
        }
        public MemoryBuffer1D<T, Stride1D.Dense> Exec<T>(IBufferOperation[] operations)
        {

        }

    }
}

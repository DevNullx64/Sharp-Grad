using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    public static class KernelProcessUnit<T>
        where T : unmanaged, INumber<T>
    {
        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="operation">Operation to perform</param>
        /// <param name="left">Left operand</param>
        /// <param name="right">Right operand</param>
        /// <param name="result">Result of the operation</param>
        private static void KPU(OpCode operation, ref T left, ref T right, ref T result)
        {
            switch (operation)
            {
                case OpCode.Load: result = left; break;
                case OpCode.Add: result += AddOp<T>.Exec(left, right); break;
                case OpCode.Sub: result += SubOp<T>.Exec(left, right); break;
                case OpCode.Mul: result += MulOp<T>.Exec(left, right); break;
                case OpCode.Div: result += DivOp<T>.Exec(left, right); break;
                default: result = T.Zero; break;
            }
        }

        /// <summary>
        /// Dynamic Kernel Processing Unit
        /// </summary>
        /// <param name="idx">GPU Index</param>
        /// <param name="ops">Operations to perform</param>
        /// <param name="left">Left operands</param>
        /// <param name="right">Right operands</param>
        /// <param name="output">Results of the operations</param>
        public static void Dynamic(Index1D idx, ArrayView<OpCode> ops, ArrayView1D<T, Stride1D.Dense> left, ArrayView1D<T, Stride1D.Dense> right, ArrayView1D<T, Stride1D.Dense> output)
        {
            for (int i = 0; i < ops.Length; i++)
                KPU(ops[i], ref left[idx], ref right[idx], ref output[idx]);
        }

        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="idx">GPU Index</param>
        /// <param name="ops">Operations to perform</param>
        /// <param name="tensors">Tensors to operate on</param>
        public static void KPU(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<T, Stride2D.DenseX> tensors)
        {
            T accumulator = T.Zero;
            for (int i = 0; i < ops.Length; i++)
            {
                OperationKPU op = ops[i];
                T left = op.Left == -1 ? accumulator : tensors[op.Left, idx];
                T right = op.Right == -1 ? accumulator : tensors[op.Right, idx];
                T result = op.Result == -1 ? accumulator : tensors[op.Result, idx];
                KPU(op.OpCode, ref left, ref right, ref result);
                if (op.Result == -1)
                    accumulator = result;
                else
                    tensors[op.Result, idx] = result;
            }
        }
    }
}
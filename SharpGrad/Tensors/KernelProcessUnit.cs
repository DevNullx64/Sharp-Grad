using ILGPU;
using ILGPU.Runtime;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    public static class KernelProcessUnit<TType, TGrad>
        where TType : unmanaged, INumber<TType>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="operation">Operation to perform</param>
        /// <param name="left">Left operand</param>
        /// <param name="right">Right operand</param>
        /// <param name="result">Result of the operation</param>
        private static void KPU(OpCode operation, ref TType left, ref TType right, ref TType result)
        {
            switch (operation)
            {
                case OpCode.Load: result = left; break;
                case OpCode.Add: result += AddOp<TType, TGrad>.ApplyCpu(left, right); break;
                case OpCode.Sub: result += SubOp<TType, TGrad>.ApplyCpu(left, right); break;
                case OpCode.Mul: result += MulOp<TType, TGrad>.ApplyCpu(left, right); break;
                case OpCode.Div: result += DivOp<TType, TGrad>.ApplyCpu(left, right); break;
                default: result = TType.Zero; break;
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
        public static void Dynamic(Index1D idx, ArrayView<OpCode> ops, ArrayView1D<TType, Stride1D.Dense> left, ArrayView1D<TType, Stride1D.Dense> right, ArrayView1D<TType, Stride1D.Dense> output)
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
        public static void KPU(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<TType, Stride2D.DenseX> tensors)
        {
            TType accumulator = TType.Zero;
            for (int i = 0; i < ops.Length; i++)
            {
                OperationKPU op = ops[i];
                TType left = op.Left == -1 ? accumulator : tensors[op.Left, idx];
                TType right = op.Right == -1 ? accumulator : tensors[op.Right, idx];
                TType result = op.Result == -1 ? accumulator : tensors[op.Result, idx];
                KPU(op.OpCode, ref left, ref right, ref result);
                if (op.Result == -1)
                    accumulator = result;
                else
                    tensors[op.Result, idx] = result;
            }
        }
    }
}
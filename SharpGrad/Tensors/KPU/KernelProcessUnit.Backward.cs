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
using System.Runtime;
using ILGPU.Runtime.Cuda;
using System.Data.SqlTypes;
using System.Data;
using System.Net.Http.Headers;
using System.Diagnostics;
using SharpGrad.Tensors.KPU;

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (T, T) Backward<T>(OpCode operation, T operand1, T operand2, T grad)
            where T : unmanaged, INumber<T>
        {
            return operation switch
            {
                OpCode.Add => AddOp<T>.Backward(operand1, operand2, grad),
                OpCode.Sub => SubOp<T>.Backward(operand1, operand2, grad),
                OpCode.Mul => MulOp<T>.Backward(operand1, operand2, grad),
                OpCode.Div => DivOp<T>.Backward(operand1, operand2, grad),
                OpCode.Neg => (NegOp<T>.Backward(operand1, grad), T.Zero),
                _ => (T.Zero, T.Zero),
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T Backward1<T>(OpCode operation, T operand1, T operand2, T grad)
            where T : unmanaged, INumber<T>
        {
            return operation switch
            {
                OpCode.Add => AddOp<T>.Backward(operand1, operand2, grad).Item1,
                OpCode.Sub => SubOp<T>.Backward(operand1, operand2, grad).Item1,
                OpCode.Mul => MulOp<T>.Backward(operand1, operand2, grad).Item1,
                OpCode.Div => DivOp<T>.Backward(operand1, operand2, grad).Item1,
                OpCode.Neg => NegOp<T>.Backward(operand1, grad),
                _ => T.Zero,
            };
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T Backward2<T>(OpCode operation, T operand1, T operand2, T grad)
            where T : unmanaged, INumber<T>
        {
            return operation switch
            {
                OpCode.Add => AddOp<T>.Backward(operand1, operand2, grad).Item1,
                OpCode.Sub => SubOp<T>.Backward(operand1, operand2, grad).Item1,
                OpCode.Mul => MulOp<T>.Backward(operand1, operand2, grad).Item1,
                OpCode.Div => DivOp<T>.Backward(operand1, operand2, grad).Item1,
                _ => T.Zero,
            };
        }
    }
}

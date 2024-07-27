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

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T BackwardLeft<T>(OpCode operation, T operand1, T operand2, T grad)
            where T : unmanaged, INumber<T>
        {
            return operation switch
            {
                OpCode.Add => AddOp<T>.BackwardLeft(operand1, operand2, grad),
                OpCode.Sub => SubOp<T>.BackwardLeft(operand1, operand2, grad),
                OpCode.Mul => MulOp<T>.BackwardLeft(operand1, operand2, grad),
                OpCode.Div => DivOp<T>.BackwardLeft(operand1, operand2, grad),
                OpCode.Neg => NegOp<T>.Backward(operand1, grad),
                _ => T.Zero,
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T BackwardRight<T>(OpCode operation, T operand1, T operand2, T grad)
            where T : unmanaged, INumber<T>
        {
            return operation switch
            {
                OpCode.Add => AddOp<T>.BackwardRight(operand1, operand2, grad),
                OpCode.Sub => SubOp<T>.BackwardRight(operand1, operand2, grad),
                OpCode.Mul => MulOp<T>.BackwardRight(operand1, operand2, grad),
                OpCode.Div => DivOp<T>.BackwardRight(operand1, operand2, grad),
                _ => T.Zero,
            };
        }
    }
}

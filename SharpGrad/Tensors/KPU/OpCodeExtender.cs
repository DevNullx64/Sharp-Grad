﻿using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors.KPU
{
    public static class OpCodeExtender
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsOperator(this OpCode opCode)
            => (opCode & OpCode.IsOperator) != 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsFunction(this OpCode opCode)
            => !opCode.IsOperator();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static OpCode GetCodeOnly(this OpCode opCode)
            => (OpCode)((short)opCode & (short)~OpCode.IsOperator);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Is(this OpCode opCode, SharedOpCode sharedOpCode)
            => opCode.IsOperator() && (short)opCode.GetCodeOnly() == (short)sharedOpCode;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Is(this OpCode opCode, SharedFuncCode sharedFuncCode)
            => opCode.IsFunction() && (short)opCode.GetCodeOnly() == (short)sharedFuncCode;
    }
}
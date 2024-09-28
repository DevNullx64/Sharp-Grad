using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public static class OpCodeExtender
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsOperator(this OpCode opCode)
            => (opCode & OpCode.IsOperator) != 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsFunction(this OpCode opCode)
            => !IsOperator(opCode);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static OpCode GetCodeOnly(this OpCode opCode)
            => (OpCode)((short)opCode & (short)~OpCode.IsOperator);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Is(this OpCode opCode, SharedOpCode sharedOpCode)
            => IsOperator(opCode) && (short)opCode.GetCodeOnly() == (short)sharedOpCode;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Is(this OpCode opCode, SharedFuncCode sharedFuncCode)
            => IsFunction(opCode) && (short)opCode.GetCodeOnly() == (short)sharedFuncCode;
    }
}
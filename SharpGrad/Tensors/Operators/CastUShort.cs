using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastUShort : ICastable<ushort>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(ushort from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(ushort from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(ushort from) => (short)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(ushort from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(ushort from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(ushort from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(ushort from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(ushort from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(ushort from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(ushort from) => from;
    }
}

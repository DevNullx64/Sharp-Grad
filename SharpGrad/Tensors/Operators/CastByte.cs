using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastByte : ICastable<byte>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(byte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(byte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(byte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(byte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(byte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(byte from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(byte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(byte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(byte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(byte from) => from;
    }
}

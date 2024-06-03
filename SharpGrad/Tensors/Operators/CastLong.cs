using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastLong : ICastable<long>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(long from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(long from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(long from) => (short)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(long from) => (int)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(long from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(long from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(long from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(long from) => (ushort)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(long from) => (uint)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(long from) => (ulong)from;
    }
}

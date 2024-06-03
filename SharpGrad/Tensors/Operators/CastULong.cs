using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastULong : ICastable<ulong>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(ulong from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(ulong from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(ulong from) => (short)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(ulong from) => (int)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(ulong from) => (long)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(ulong from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(ulong from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(ulong from) => (ushort)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(ulong from) => (uint)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(ulong from) => from;
    }
}

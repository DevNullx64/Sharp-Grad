using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastUInt : ICastable<uint>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(uint from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(uint from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(uint from) => (short)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(uint from) => (int)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(uint from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(uint from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(uint from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(uint from) => (ushort)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(uint from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(uint from) => (ulong)from;
    }
}

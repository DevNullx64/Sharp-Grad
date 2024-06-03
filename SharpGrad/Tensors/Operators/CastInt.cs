using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastInt : ICastable<int>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(int from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(int from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(int from) => (short)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(int from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(int from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(int from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(int from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(int from) => (ushort)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(int from) => (uint)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(int from) => (ulong)from;
    }
}

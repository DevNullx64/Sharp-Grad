using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastDouble : ICastable<double>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(double from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(double from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(double from) => (short)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(double from) => (int)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(double from) => (long)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(double from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(double from) => (float)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(double from) => (ushort)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(double from) => (uint)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(double from) => (ulong)from;
    }
}

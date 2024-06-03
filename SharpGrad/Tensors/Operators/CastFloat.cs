using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastFloat : ICastable<float>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(float from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(float from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(float from) => (short)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(float from) => (int)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(float from) => (long)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(float from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(float from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(float from) => (ushort)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(float from) => (uint)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(float from) => (ulong)from;
    }
}

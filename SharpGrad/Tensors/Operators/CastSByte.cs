using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastSByte : ICastable<sbyte>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(sbyte from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(sbyte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(sbyte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(sbyte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(sbyte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(sbyte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(sbyte from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(sbyte from) => (ushort)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(sbyte from) => (uint)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(sbyte from) => (ulong)from;
    }
}

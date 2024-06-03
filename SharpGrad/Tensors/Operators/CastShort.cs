using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public class CastShort : ICastable<short>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(short from) => (byte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(short from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(short from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(short from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(short from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static sbyte ToSByte(short from) => (sbyte)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(short from) => from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(short from) => (ushort)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(short from) => (uint)from;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(short from) => (ulong)from;
    }
}

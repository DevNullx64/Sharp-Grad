using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public interface ICastable<T>
        where T : unmanaged, INumber<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static double ToDouble(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static float ToSingle(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static long ToInt64(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static ulong ToUInt64(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static int ToInt32(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static uint ToUInt32(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static short ToInt16(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static ushort ToUInt16(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static sbyte ToSByte(T from);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        abstract static byte ToByte(T from);
    }
}

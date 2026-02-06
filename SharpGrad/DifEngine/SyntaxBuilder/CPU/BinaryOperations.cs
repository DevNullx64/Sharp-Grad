//#define MP
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
#if MP
using System.Threading.Tasks;
#endif

namespace SharpGrad.DifEngine.SyntaxBuilder.CPU
{
    public static class BinaryOperations
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AddForward<T>(T left, T right)
            where T : INumber<T>
            => left + right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AddBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AddBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SubtractForward<T>(T left, T right)
            where T : INumber<T>
            => left - right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SubtractBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SubtractBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => -gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MultiplyForward<T>(T left, T right)
            where T : INumber<T>
            => left * right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MultiplyBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => right * gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MultiplyBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => left * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T DivideForward<T>(T left, T right)
            where T : INumber<T>
            => left / right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T DivideBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput / right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T DivideBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => -left * gradOutput / (right * right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T PowForward<T>(T left, T right)
            where T : INumber<T>, IPowerFunctions<T>
            => T.Pow(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T PowBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IPowerFunctions<T>
            => right * T.Pow(left, right - T.One) * gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T PowBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
            => T.Log(left) * T.Pow(left, right) * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ModuloForward<T>(T left, T right)
            where T : INumber<T>
            => left % right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ModuloBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ModuloBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => -(left / right) * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MaxForward<T>(T left, T right)
            where T : INumber<T>
            => T.Max(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MaxBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IComparable<T>
            => T.CreateTruncating(left.CompareTo(right) >= 0 ? gradOutput : T.Zero);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MaxBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IComparable<T>
            => T.CreateTruncating(right.CompareTo(left) > 0 ? gradOutput : T.Zero);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MinForward<T>(T left, T right)
            where T : INumber<T>
            => T.Min(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MinBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IComparable<T>
            => T.CreateTruncating(left.CompareTo(right) <= 0 ? gradOutput : T.Zero);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MinBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IComparable<T>
            => T.CreateTruncating(right.CompareTo(left) < 0 ? gradOutput : T.Zero);
    }
}
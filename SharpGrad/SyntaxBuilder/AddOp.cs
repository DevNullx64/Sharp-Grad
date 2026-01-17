using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.SyntaxBuilder
{
    public readonly struct AddOp<T> : IBinaryOperation<T>
            where T : INumber<T>, IAdditionOperators<T, T, T>
        {
            public static T Identity
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => T.Zero;
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T Operate(T a, T b) => a + b;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static (G Left, G Right) Backward<G>(T left, T right, G gradOut) where G : IFloatingPoint<G>
            {
                return (gradOut, gradOut);
            }
        }
}
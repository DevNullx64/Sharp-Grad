using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.SyntaxBuilder
{
    public readonly struct DivOp<T> : IBinaryOperation<T>
            where T : INumber<T>, IDivisionOperators<T, T, T>
        {
            public static T Identity
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => T.One;
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T Operate(T a, T b) => a / b;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static (G Left, G Right) Backward<G>(T left, T right, G gradOut) where G : IFloatingPoint<G>
            {
                G l = G.CreateChecked(left);
                G r = G.CreateChecked(right);
                return (
                    gradOut / r,
                    -gradOut * l / (r * r)
                );
            }
        }
}
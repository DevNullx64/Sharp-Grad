using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder.Functions
{
    public readonly struct ReLU<T>
        where T : struct, INumber<T>, IComparable<T>
    {
        public string Name
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => "ReLU";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Operate(T input)
            => input < T.Zero ? T.Zero : input;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public G Backward<G>(T input, T result, G grad) 
            where G : struct, IBinaryFloatingPointIeee754<G>
        {
            // Utilise result au lieu de recalculer
            // result == T.Zero si input < T.Zero, sinon result == input
            return G.CreateChecked(result) > G.Zero ? grad : G.Zero;
        }
    }
}

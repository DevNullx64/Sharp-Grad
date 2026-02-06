using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder.Functions
{
    public readonly struct LeakyReLU<T>
        where T : struct, INumber<T>
    {
        public string Name
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => "LeakyReLU";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Operate(T input, T alpha)
            => input > T.Zero ? input : alpha * input;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public (G Left, G Right) Backward<G>(T input, T alpha, T result, G grad) 
            where G : struct, IBinaryFloatingPointIeee754<G>
        {
            if (input > T.Zero)
                return (grad, G.Zero);
            else
                return (
                    grad * G.CreateChecked(alpha),
                    grad * G.CreateChecked(input)
                );
        }
    }
}

using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder.Functions
{
    public readonly struct Tanh<T>
        where T : struct, IFloatingPoint<T>, IHyperbolicFunctions<T>
    {
        public string Name
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => "tanh";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Operate(T input) => T.Tanh(input);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public G Backward<G>(T input, T result, G grad) 
            where G : struct, IBinaryFloatingPointIeee754<G>
        {
            // Utilise result (déjà calculé) au lieu de recalculer T.Tanh(input)
            // d/dx(tanh(x)) = 1 - tanh²(x)
            G tanhValue = G.CreateChecked(result);
            G derivative = G.One - tanhValue * tanhValue;
            return grad * derivative;
        }
    }
}

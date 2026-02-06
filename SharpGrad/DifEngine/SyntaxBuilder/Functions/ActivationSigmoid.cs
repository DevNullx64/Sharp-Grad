using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder.Functions
{
    public readonly struct Sigmoid<T>
        where T : struct, IFloatingPoint<T>, IExponentialFunctions<T>
    {
        public string Name
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => "sigmoid";
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Operate(T input)
            => T.One / (T.One + T.Exp(-input));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public G Backward<G>(T input, T result, G grad) 
            where G : struct, IBinaryFloatingPointIeee754<G>
        {
            // Utilise result (déjà calculé) au lieu de recalculer sigmoid
            // d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
            G sigmoid = G.CreateChecked(result);
            G derivative = sigmoid * (G.One - sigmoid);
            return grad * derivative;
        }
    }
}

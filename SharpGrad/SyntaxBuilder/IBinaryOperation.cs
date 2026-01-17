using System.Numerics;

namespace SharpGrad.SyntaxBuilder
{
    public interface IBinaryOperation<T> : IOperation<T>
            where T : INumber<T>
        {
            static abstract T Identity { get; }
            static abstract T Operate(T left, T right);
            static abstract (G Left, G Right) Backward<G>(T left, T right, G gradOut) where G : IFloatingPoint<G>;
        }
}
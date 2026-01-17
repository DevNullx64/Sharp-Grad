using System.Numerics;

namespace SharpGrad.SyntaxBuilder
{
    public interface IUnaryOperation<T> : IOperation<T>
            where T : INumber<T>
        {
            static abstract T Operate(T a);
            static abstract G Backward<G>(T a, G gradA) where G : IFloatingPoint<G>;
        }
}
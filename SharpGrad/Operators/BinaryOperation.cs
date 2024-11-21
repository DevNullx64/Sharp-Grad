using SharpGrad.DifEngine;
using System.Numerics;

namespace SharpGrad.Operators
{
    public class BinaryOperation<TOp, TType>(Value<TType> left, Value<TType> right)
        : Value<TType>(TOp.Invoke(left.Data, right.Data), TOp.Symbol, left, right)
        where TOp : IExecBinary<TType, TType, TType>
        where TType : unmanaged, INumber<TType>
    {
        protected override void Backward()
        {
            var (gradLeft, gradRight) = TOp.Backward(LeftChildren!.Data, RightChildren!.Data, Grad);
            LeftChildren!.Grad += gradLeft;
            RightChildren!.Grad += gradRight;
        }
    }
}
using SharpGrad.DifEngine;
using SharpGrad.Operators;
using System.Numerics;

namespace SharpGrad
{
    public class UnaryOperation<TOp, TType>(Value<TType> value)
        : Value<TType>(TOp.Invoke(value.Data), TOp.Symbol, value)
        where TOp : IExecUnary<TType, TType>
        where TType : unmanaged, INumber<TType>
    {
        protected override void Backward()
        {
            LeftChildren!.Grad += TOp.Backward(LeftChildren!.Data, Grad);
        }
    }
}
using System;
using System.Numerics;

namespace SharpGrad.DifEngine
{
    public class PowValue<TType>(Value<TType> left, Value<TType> right)
        : Value<TType>(TType.Pow(left.Data, right.Data), "^", left, right)
        where TType : unmanaged, INumber<TType>, IPowerFunctions<TType>, ILogarithmicFunctions<TType>
    {
        protected override void Backward()
        {
            LeftChildren.Grad += Grad * RightChildren.Data * TType.Pow(LeftChildren.Data, RightChildren.Data - TType.One);
            RightChildren.Grad += Grad * TType.Pow(LeftChildren.Data, RightChildren.Data) * TType.Log(LeftChildren.Data);
        }
    }
}
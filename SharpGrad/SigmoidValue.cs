using System.Numerics;

namespace SharpGrad.DifEngine
{
    public class SigmoidValue<TType>(Value<TType> value) : Value<TType>(TType.One / (TType.One + TType.Exp(-value.Data)), "sigmoid", value)
        where TType : unmanaged, INumber<TType>, IExponentialFunctions<TType>
    {
        protected override void Backward()
        {
            var sigmoid = Data;
            LeftChildren.Grad += Grad * sigmoid * (TType.One - sigmoid);
        }
    }
}

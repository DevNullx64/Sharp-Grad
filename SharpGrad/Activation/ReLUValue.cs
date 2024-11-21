using System.Numerics;
using SharpGrad.DifEngine;

namespace SharpGrad.Activation
{
    public class ReLUValue<TType> : Value<TType>
        where TType : unmanaged, INumber<TType>
    {
        public ReLUValue(Value<TType> value)
            : base(value.Data <= TType.Zero ? TType.Zero : value.Data, "relu", value)
        {
        }

        protected override void Backward()
        {
            if (Grad > TType.Zero)
                LeftChildren.Grad += Grad;
        }
    }
}
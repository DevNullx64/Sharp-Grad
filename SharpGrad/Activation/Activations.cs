using SharpGrad.DifEngine;
using System.Numerics;

namespace SharpGrad.Activation
{
    public static class Activations
    {
        public static Value<TType> ReLU<TType>(Value<TType> value)
            where TType : unmanaged, INumber<TType>
            => new ReLUValue<TType>(value);

        public static Value<TType> LeakyReLU<TType>(Value<TType> value, TType alpha)
            where TType : unmanaged, INumber<TType>
            => new LeakyReLUValue<TType>(value, alpha);

        public static Value<TType> Sigmoid<TType>(Value<TType> value)
            where TType : unmanaged, INumber<TType>, IExponentialFunctions<TType>
            => new SigmoidValue<TType>(value);
    }
}
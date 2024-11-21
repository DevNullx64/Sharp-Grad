using System.Numerics;

namespace SharpGrad.DifEngine
{
    public static class DMath
    {
        public static Value<T> Pow<T>(Value<T> left, Value<T> right)
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
            => new PowValue<T>(left, right);
        public static Value<T> Tanh<T>(Value<T> @this)
            where T : unmanaged, INumber<T>, IHyperbolicFunctions<T>
            => new TanhValue<T>(@this);

        public static Value<T> Sigmoid<T>(Value<T> @this)
            where T : unmanaged, INumber<T>, IExponentialFunctions<T>
            => new SigmoidValue<T>(@this);
    }
}
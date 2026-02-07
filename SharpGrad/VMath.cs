using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    public static class VMath
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> Pow<TType>(this Value<TType> @this, Value<TType> exponent)
            where TType : struct, IBinaryFloatingPointIeee754<TType>, IPowerFunctions<TType>
            => new(KindBinary.Power, @this, exponent);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ReducedValue<TType> Sum<TType>(this Value<TType> @this, params Dimension[] toReduce)
            where TType : struct, INumber<TType>
            => new(KindReduction.Sum, @this, toReduce);
    }
}
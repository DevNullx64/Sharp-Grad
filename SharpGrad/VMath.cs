using SharpGrad.DifEngine.SyntaxBuilder;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    public static class VMath
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UnaryComputedValue<TType> Exp<TType>(this Value<TType> @this)
            where TType : struct, INumber<TType>, IExponentialFunctions<TType>
            => new(KindUnary.Exp, @this);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> Pow<TType>(this Value<TType> @this, Value<TType> exponent)
            where TType : struct, INumber<TType>, IPowerFunctions<TType>
            => new(KindBinary.Power, @this, exponent);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Value<TType> Sum<TType>(this Value<TType> @this, params Dimension[] toReduce)
            where TType : struct, INumber<TType>
        {
            // toReduce = [.. toReduce.Where(d => !d.IsScalar).Distinct()];
            List<Dimension> toReduceList = [];
            foreach (Dimension d in toReduce)
            {
                if (!d.IsScalar && !toReduceList.Contains(d))
                    toReduceList.Add(d);
            }
            return toReduceList.Count == 0
                ? @this
                : new ReducedValue<TType>(KindReduction.Sum, @this, toReduceList.ToArray());
        }
    }
}
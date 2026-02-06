using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad
{
    public static class CollectionExtender
    {
        public static (T Sum, T Count) GetSumCount<T>(this IEnumerable<T> @this)
            where T : INumber<T>
        {
            T sum = T.AdditiveIdentity;
            T count = T.Zero;
            foreach (var item in @this)
            {
                sum += item;
                count++;
            }
            return (sum, count);
        }
        public static T Sum<T>(this IEnumerable<T> @this)
            where T : INumber<T> => @this.GetSumCount().Sum;
        public static T Sum<T>(this T[] @this)
            where T : INumber<T> => @this.AsEnumerable().Sum();

        public static (T Mean, T Sum, T Count) GetMeanSumCount<T>(this IEnumerable<T> @this)
            where T : INumber<T>
        {
            (T sum, T count) = @this.GetSumCount();
            return (sum / count, sum, count);
        }
        public static T Mean<T>(this IEnumerable<T> @this)
            where T : INumber<T> => @this.GetMeanSumCount().Mean;
        public static T Mean<T>(this T[] @this)
            where T : INumber<T> => @this.AsEnumerable().Mean();


        public static (T Var, T mean, T Sum, T Count) GetVarMeanSumCount<T>(this IEnumerable<T> @this)
            where T : INumber<T>, IRootFunctions<T>
        {
            (T mean, T sum, T count) = @this.GetMeanSumCount();
            T sumOfSquares = T.Zero;
            foreach (var item in @this)
            {
                T diff = item - mean;
                sumOfSquares += diff * diff;
            }
            return (sumOfSquares / count, mean, sum, count);
        }

        public static T Var<T>(this IEnumerable<T> @this)
            where T : INumber<T>, IRootFunctions<T> => @this.GetVarMeanSumCount().Var;
        public static T Var<T>(this T[] @this)
            where T : INumber<T>, IRootFunctions<T> => @this.AsEnumerable().Var();

        public static (T Std, T Var, T mean, T Sum, T Count) GetStdVarMeanSumCount<T>(this IEnumerable<T> @this)
            where T : INumber<T>, IRootFunctions<T>
        {
            (T var, T mean, T sum, T count) = @this.GetVarMeanSumCount();
            return (T.Sqrt(var), var, mean, sum, count);
        }
        public static T Std<T>(this IEnumerable<T> @this)
            where T : INumber<T>, IRootFunctions<T>
            => T.Sqrt(@this.Var());
        public static T Std<T>(this T[] @this)
            where T : INumber<T>, IRootFunctions<T>
            => T.Sqrt(@this.Var());
    }
}
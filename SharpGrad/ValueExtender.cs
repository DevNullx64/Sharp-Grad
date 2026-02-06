using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad
{
    public static class ValueExtender
    {
        public static bool IsOutput<TType>(this IEnumerable<Value<TType>> @this) 
            where TType : struct, INumber<TType>
        {
            return @this.All(v => v.IsOutput);
        }
        
        public static bool IsOutput<TType>(this Value<TType>[] @this) 
            where TType : struct, INumber<TType>
            => @this.AsEnumerable().IsOutput();

        public static void IsOutput<TType>(this IEnumerable<Value<TType>> @this, bool value) 
            where TType : struct, INumber<TType>
        {
            foreach (var v in @this)
            {
                v.IsOutput = value;
            }
        }
        
        public static void IsOutput<TType>(this Value<TType>[] @this, bool value) 
            where TType : struct, INumber<TType>
            => @this.AsEnumerable().IsOutput(value);
    }
}

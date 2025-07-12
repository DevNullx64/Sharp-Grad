using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpGrad.DifEngine
{
    public static class DimensionExtender
    {
        public static string GetString<T>(this IEnumerable<T> arr) => '[' + string.Join(", ", arr) + ']';
        public static string GetString(this Dimension[] dim) => GetString(dim.Select(d => d.Size));

        public static int Size(this Dimension[] @this)
        {
            int size = 1;
            for (int i = 0; i < @this.Length; i++)
            {
                checked
                {
                    size *= @this[i].Size;
                }
            }
            return size;
        }

        public static bool IsScalar(this Dimension[] @this)
            => @this.Length == 0;

        public static bool IsVector(this Dimension[] @this)
            => @this.Length == 1;

        public static int GetLinearIndex(this Dimension[] shape, int[] indices)
        {
            if (shape.Length == 0 && indices.Length == 1)
                return 0;
            if (shape.Length != indices.Length)
            {
                throw new ArgumentException($"The shape size {shape.Size()} is not equal to the indices length {indices.Length}");
            }
            int index = 0;
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                checked
                {
                    index += indices[i] * stride;
                    stride *= shape[i].Size;
                }
            }
            return index;
        }
    }
}

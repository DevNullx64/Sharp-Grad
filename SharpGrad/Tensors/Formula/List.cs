using System;
using System.Collections.Generic;

namespace SharpGrad.Tensors.Formula
{
    public class List<TSource, TOutput>(Func<int, TSource, TOutput> convert) : List<TSource>, IReadOnlyList<TOutput>
    {
        public new TOutput this[int index]
            => convert(index, base[index]);

        IEnumerator<TOutput> IEnumerable<TOutput>.GetEnumerator()
        {
            for (int i = 0; i < Count; i++)
                yield return convert(i, base[i]);
        }
    }
}
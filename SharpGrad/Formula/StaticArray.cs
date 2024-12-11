using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Formula
{
    public interface IStaticArray<T> : IReadOnlyList<T>
    { }

    public readonly struct StaticArray<T>() : IStaticArray<T>
        where T: unmanaged, IEquatable<T>
    {
        public readonly int Count => 0;
        public readonly T this[int index] => throw new IndexOutOfRangeException();
        public readonly IEnumerator<T> GetEnumerator() => Enumerable.Empty<T>().GetEnumerator();
        readonly IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    public struct StaticArray1<T>(T item0 = default) : IStaticArray<T>
        where T : unmanaged, IEquatable<T>
    {
        public T Item0 = item0;
        public readonly int Count => Item0.Equals(default) ? 0 : 1;
        public T this[int index]
        {
            readonly get => index == 0
                ? (Item0.Equals(default) ? throw new IndexOutOfRangeException() : Item0)
                : throw new IndexOutOfRangeException();
            set => Item0 = index == 0 ? value : throw new IndexOutOfRangeException();
        }

        public readonly IEnumerator<T> GetEnumerator()
        {
            if (!Item0.Equals(default))
                yield return Item0;
        }
        readonly IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    public readonly struct StaticArray2<T>(T item0, T item1) : IStaticArray<T>
    {
        public readonly T Item0 = item0;
        public readonly T Item1 = item1;
        public readonly int Count => 2;
        public readonly T this[int index]
            => index == 0 ? Item0 : index == 1 ? Item1 : throw new IndexOutOfRangeException();

        public readonly IEnumerator<T> GetEnumerator()
        {
            yield return Item0;
            yield return Item1;
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}

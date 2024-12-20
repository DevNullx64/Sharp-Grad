using System;

namespace SharpGrad.Formula.Internal
{
    public interface IInternalStaticReadOnlyArray<T, TXD>
        where T : unmanaged
        where TXD : IXD
    {
        int Count { get; }
        T this[int index] { get; }
        public int IndexOf(T value);
    }

    internal readonly struct InternalStaticReadOnlyArray1<T>(T index0) : IInternalStaticReadOnlyArray<T, _1D>
        where T : unmanaged
    {
        private readonly InternalStaticArray1<T> Array = new(index0);
        public readonly int Count => Array.Count;
        public readonly T this[int index] => Array[index];

        public int IndexOf(T value)
        {
            if (Array.Index0.Equals(value))
                return 0;
            return -1;
        }
    }

    internal readonly struct InternalStaticReadOnlyArray2<T>(T index0, T index1) : IInternalStaticReadOnlyArray<T, _2D>
        where T : unmanaged
    {
        private readonly InternalStaticArray2<T> Array = new(index0, index1);
        public readonly int Count => Array.Count;
        public readonly T this[int index] => Array[index];

        public int IndexOf(T value)
        {
            if (Array.Index0.Equals(value))
                return 0;
            if (Array.Index1.Equals(value))
                return 1;
            return -1;
        }
    }

    internal readonly struct InternalStaticReadOnlyArray3<T>(T index0, T index1, T index2) : IInternalStaticReadOnlyArray<T, _3D>
        where T : unmanaged
    {
        private readonly InternalStaticArray3<T> Array = new(index0, index1, index2);
        public readonly int Count => Array.Count;
        public readonly T this[int index] => Array[index];

        public int IndexOf(T value)
        {
            if (Array.Index0.Equals(value))
                return 0;
            if (Array.Index1.Equals(value))
                return 1;
            if (Array.Index2.Equals(value))
                return 2;
            return -1;
        }
    }
}

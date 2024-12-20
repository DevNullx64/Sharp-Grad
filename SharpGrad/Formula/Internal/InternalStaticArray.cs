using System;

namespace SharpGrad.Formula.Internal
{
    public interface IInternalStaticArray<T, TXD> : IInternalStaticReadOnlyArray<T, TXD>
        where T : unmanaged
        where TXD : IXD
    {
        new T this[int index] { get; set; }
    }

    internal struct InternalStaticArray1<T>(T index0) : IInternalStaticArray<T, _1D>
        where T : unmanaged
    {
        public T Index0 = index0;
        public readonly int Count { get; } = 1;
        public T this[int index]
        {
            readonly get
            {
                if (index < Count)
                {
                    if (index == 0)
                        return Index0;
                }
                throw new IndexOutOfRangeException();
            }
            set
            {
                if (index < Count)
                {
                    if (index == 0)
                        Index0 = value;
                }
                throw new IndexOutOfRangeException();
            }
        }

        public int IndexOf(T value)
        {
            if (Index0.Equals(value))
                return 0;
            return -1;
        }
    }

    internal struct InternalStaticArray2<T>(T index0, T index1) : IInternalStaticArray<T, _2D>
        where T : unmanaged
    {
        public T Index0 = index0;
        public T Index1 = index1;
        public readonly int Count { get; } = 2;
        public T this[int index]
        {
            readonly get
            {
                if (index < Count)
                {
                    switch (index)
                    {
                        case 0: return Index0;
                        case 1: return Index1;
                    }
                }
                throw new IndexOutOfRangeException();
            }
            set
            {
                if (index < Count)
                {
                    switch (index)
                    {
                        case 0: Index0 = value; break;
                        case 1: Index1 = value; break;
                    }
                }
                throw new IndexOutOfRangeException();
            }
        }

        public int IndexOf(T value)
        {
            if (Index0.Equals(value))
                return 0;
            if (Index1.Equals(value))
                return 1;
            return -1;
        }
    }

    internal struct InternalStaticArray3<T>(T index0, T index1, T index2) : IInternalStaticArray<T, _3D>
        where T : unmanaged
    {
        public T Index0 = index0;
        public T Index1 = index1;
        public T Index2 = index2;
        public readonly int Count { get; } = 3;
        public T this[int index]
        {
            readonly get
            {
                if (index < Count)
                {
                    switch (index)
                    {
                        case 0: return Index0;
                        case 1: return Index1;
                        case 2: return Index2;
                    }
                }
                throw new IndexOutOfRangeException();
            }
            set
            {
                if (index < Count)
                {
                    switch (index)
                    {
                        case 0: Index0 = value; break;
                        case 1: Index1 = value; break;
                        case 2: Index2 = value; break;
                    }
                }
                throw new IndexOutOfRangeException();
            }
        }

        public int IndexOf(T value)
        {
            if (Index0.Equals(value))
                return 0;
            if (Index1.Equals(value))
                return 1;
            if (Index2.Equals(value))
                return 2;
            return -1;
        }
    }
}

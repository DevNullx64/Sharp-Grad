using System;

namespace SharpGrad.Formula.Internal
{
    internal interface IInternalIndices<TXD>
        where TXD : IXD
    {
        int Count { get; }
        int this[int index] { get; set; }
    }

    internal struct InternalIndices1(int index0) : IInternalIndices<_1D>
    {
        public int Index0 = index0;

        public readonly int Count { get; } = index0 < 0 ? 0 : 1;

        public int this[int index]
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
    }

    internal struct InternalIndices2(int index0, int index1) : IInternalIndices<_2D>
    {
        public int Index0 = index0;
        public int Index1 = index1;

        public readonly int Count { get; } = index0 < 0 ? 0 : index1 < 0 ? 1 : 2;

        public int this[int index]
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
    }

    internal struct InternalIndices3(int index0, int index1, int index2) : IInternalIndices<_3D>
    {
        public int Index0 = index0;
        public int Index1 = index1;
        public int Index2 = index2;
        public readonly int Count { get; } = index0 < 0 ? 0 : index1 < 0 ? 1 : index2 < 0 ? 2 : 3;
        public int this[int index]
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
    }
}

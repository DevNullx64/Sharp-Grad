using ILGPU;
using ILGPU.Runtime;
using System;

namespace SharpGrad.Formula.Internal
{
    public interface IInternalShape<TXD>
        where TXD : IXD
    {
        byte Rank { get; }
        byte this[int index] { get; }

        int IndexOf(byte dimIdx);
    }

    internal readonly struct InternalShape1(byte dimensionIdx) : IInternalShape<_1D>
    {
        public readonly byte DimensionIdx0 = dimensionIdx;

        public readonly byte Rank { get; } = (byte)(dimensionIdx == default ? 0 : 1);

        public byte this[int index]
        {
            get
            {
                if (index == 0)
                    return DimensionIdx0;
                throw new IndexOutOfRangeException();
            }
        }

        public int IndexOf(byte dimIdx)
        {
            if (dimIdx == DimensionIdx0)
                return 0;
            return -1;
        }
    }

    internal readonly struct InternalShape2(byte dimensionIdx0, byte dimensionIdx1) : IInternalShape<_2D>
    {
        public readonly byte DimensionIdx0 = dimensionIdx0;
        public readonly byte DimensionIdx1 = dimensionIdx1;

        public readonly byte Rank { get; } = (byte)(dimensionIdx0 == default ? 0 : dimensionIdx1 == default ? 1 : 2);

        public byte this[int index]
        {
            get
            {
                if (index < Rank)
                {
                    switch (index)
                    {
                        case 0: return DimensionIdx0;
                        case 1: return DimensionIdx1;
                    }
                }
                throw new IndexOutOfRangeException();
            }
        }

        public int IndexOf(byte dimIdx)
        {
            if (dimIdx == DimensionIdx0)
                return 0;
            if (dimIdx == DimensionIdx1)
                return 1;
            return -1;
        }
    }

    internal readonly struct InternalShape3(byte dimensionIdx0, byte dimensionIdx1, byte dimensionIdx2) : IInternalShape<_3D>
    {
        public readonly byte DimensionIdx0 = dimensionIdx0;
        public readonly byte DimensionIdx1 = dimensionIdx1;
        public readonly byte DimensionIdx2 = dimensionIdx2;

        public readonly byte Rank { get; } = (byte)(dimensionIdx0 == default ? 0 : dimensionIdx1 == default ? 1 : dimensionIdx2 == default ? 2 : 3);

        public byte this[int index]
        {
            get
            {
                if (index < Rank)
                {
                    switch (index)
                    {
                        case 0: return DimensionIdx0;
                        case 1: return DimensionIdx1;
                        case 2: return DimensionIdx2;
                    }
                }
                throw new IndexOutOfRangeException();
            }
        }

        public int IndexOf(byte dimIdx)
        {
            if (dimIdx == DimensionIdx0)
                return 0;
            if (dimIdx == DimensionIdx1)
                return 1;
            if (dimIdx == DimensionIdx2)
                return 2;
            return -1;
        }
    }
}

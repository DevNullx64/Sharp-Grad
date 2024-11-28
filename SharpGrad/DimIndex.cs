using System;

namespace SharpGrad
{
    public class DimIndex
    {
        public Dimension Dimention { get; }
        public int Index { get; }

        public DimIndex(Dimension dimension, Index index)
        {
            Dimention = dimension;
            Index = index.GetOffset(Dimention.Size);
            if (Index < 0 || Index >= dimension.Size)
                throw new ArgumentOutOfRangeException(nameof(index), $"The index must be between 0 and {dimension.Size - 1}. Got {index.Value}.");
        }
    }
}
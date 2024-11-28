using System;

namespace SharpGrad
{
    public class DimensionalIndex
    {
        public Dimension Dimension { get; }
        public int Index { get; }

        public DimensionalIndex
            (Dimension dimension, Index index)
        {
            Dimension = dimension;
            Index = index.GetOffset(Dimension.Size);
            if (Index < 0 || Index >= dimension.Size)
                throw new ArgumentOutOfRangeException(nameof(index), $"The index must be between 0 and {dimension.Size - 1}. Got {index.Value}.");
        }
    }
}
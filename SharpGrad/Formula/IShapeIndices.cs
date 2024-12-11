namespace SharpGrad.Formula
{
    public interface IShapeIndices
    {
        Shape Shape { get; }
        int[] Offsets { get; }

        int GetDimensionOffset(Dimension dimension);
        void SetDimensionOffset(Dimension dimension, int offset);
    }
}
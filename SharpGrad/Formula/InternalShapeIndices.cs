using ILGPU;
using ILGPU.Runtime;

namespace SharpGrad.Formula
{
    internal readonly struct InternalShapeIndices<TShape, TIndices, TXD>(TShape shape, TIndices indices) where TShape : unmanaged, IInternalShape<TXD>
        where TIndices : unmanaged, IInternalIndices<TXD>
        where TXD : IXD
    {
        private readonly TShape shape = shape.Rank <= indices.Count ? shape : throw new System.ArgumentException("Shape rank must be less than or equal to indices count.");
        private readonly TIndices indices = indices;

        public readonly TShape Shape => shape;

        public readonly TIndices Indices => indices;

        public int this[byte dimIndex]
        {
            get
            {
                for (int d = 0; d < shape.Rank; d++)
                {
                    if (shape[d] == dimIndex)
                        return indices[d];
                }
                throw new System.ArgumentException("Dimension index not found in shape.");
            }
        }

        public readonly long GetIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions)
            => GetIndex(shape, indices, dimensions);

        public InternalShapeIndices<TShape, TIndices, TXD> FromShapeIndices(InternalShapeIndices<TShape, TIndices, TXD> internalShapeIndices)
            => FromShapeIndices(shape, internalShapeIndices);

        public static InternalShapeIndices<TShape, TIndices, TXD> FromShapeIndices(TShape shape, InternalShapeIndices<TShape, TIndices, TXD> otherShape)
        {
            TIndices indices = default;
            for (int d = 0; d < shape.Rank; d++)
            {
                indices[d] = otherShape[shape[d]];
            }
            return new InternalShapeIndices<TShape, TIndices, TXD>(shape, indices);
        }

        public static long GetIndex(TShape shape, TIndices indices, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions)
        {
            long index = 0;
            for (int d = 0; d < shape.Rank; d++)
            {
                int size = dimensions[shape[d]].Size;
                index *= size;
                index += indices[d];
            }
            return index;
        }

        public static InternalShapeIndices<TShape, TIndices, TXD> GetIndices(TShape shape, long index, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions)
        {
            TIndices indices = default;
            for (int d = shape.Rank - 1; d >= 0; d--)
            {
                int size = dimensions[shape[d]].Size;
                indices[d] = (int)(index % size);
                index /= size;
            }
            return new InternalShapeIndices<TShape, TIndices, TXD>(shape, indices);
        }
    }
}

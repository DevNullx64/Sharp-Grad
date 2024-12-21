using ILGPU;
using ILGPU.Runtime;

namespace SharpGrad.Formula.Internal
{
    internal readonly struct InternalShapeIndices<TShape, TIndices, TXD>(TShape shape, TIndices indices)
        where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
        where TIndices : unmanaged, IInternalStaticArray<int, TXD>
        where TXD : IXD
    {
        public readonly TShape Shape => shape;

        public readonly TIndices Indices => indices;

        public int this[BIndex<byte> dimIndex]
        {
            get
            {
                if (!dimIndex.IsEmpty)
                {
                    for (int d = 0; d < shape.Count; d++)
                    {
                        if (shape[d] == dimIndex)
                            return indices[d];
                    }
                }
                return -1;
            }
        }

        public static InternalShapeIndices<TShape, TIndices, TXD> MapIndicesFromShape(TShape shape, InternalShapeIndices<TShape, TIndices, TXD> otherShape)
        {
            TIndices indices = default;
            for (int d = 0; d < shape.Count; d++)
            {
                BIndex<byte> dimIndex = shape[d];
                if(dimIndex.IsEmpty)
                    throw new System.ArgumentException($"Shape index {d} is empty.");

                indices[d] = otherShape[dimIndex];
            }
            return new InternalShapeIndices<TShape, TIndices, TXD>(shape, indices);
        }

        public InternalShapeIndices<TShape, TIndices, TXD> MapIndicesFromShape(InternalShapeIndices<TShape, TIndices, TXD> internalShapeIndices)
            => MapIndicesFromShape(shape, internalShapeIndices);

        public static long GetIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TShape shape, TIndices indices)
        {
            long index = 0;
            for (int d = 0; d < shape.Count; d++)
            {
                BIndex<byte> dimIndex = shape[d];
                if (!dimIndex.IsEmpty)
                {
                    index *= dimensions[dimIndex].Size;
                    index += indices[d];
                }
            }
            return index;
        }

        public static InternalShapeIndices<TShape, TIndices, TXD> GetIndices(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TShape shape, long index)
        {
            TIndices indices = default;
            for (int d = shape.Count - 1; d >= 0; d--)
            {
                BIndex<byte> dimIndex = shape[d];
                if (dimIndex.IsEmpty)
                {
                    indices[d] = 0;
                }
                else
                {
                    int size = dimensions[shape[d]].Size;
                    indices[d] = (int)(index % size);
                    index /= size;
                }
            }
            return new InternalShapeIndices<TShape, TIndices, TXD>(shape, indices);
        }
    }
}

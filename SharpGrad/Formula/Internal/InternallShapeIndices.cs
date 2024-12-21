using ILGPU.Runtime;
using ILGPU;

namespace SharpGrad.Formula.Internal
{
    internal readonly struct InternallShapeIndices<TShape, TIndices, TXD>(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte shapeIdx, TIndices indices)
        where TShape : unmanaged, IInternalDimensionIndexList<TXD>
        where TIndices : unmanaged, IInternalStaticArray<int, TXD>
        where TXD : IXD
    {
        public static TIndices GetIndices(TIndices sizes, long offset)
        {
            TIndices indices = default;
            for (int d = 0; d < sizes.Count; d++)
            {
                indices[d] = (int)(offset % sizes[d]);
                offset /= sizes[d];
            }
            return indices;
        }

        public readonly InternalShape<TShape, TIndices, TXD> Shape = new InternalShape<TShape, TIndices, TXD>(shapes, dimensions, shapeIdx);
        public readonly TIndices Indices = indices;

        public InternallShapeIndices(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte shapeIdx, long offset)
            : this(shapes, dimensions, shapeIdx, GetIndices(new InternalShape<TShape, TIndices, TXD>(shapes, dimensions, shapeIdx).Sizes, offset))
        { }
    }
}

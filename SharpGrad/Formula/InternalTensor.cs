using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Formula
{
    internal interface IInternalTensor<TShape, TIndices, TXD>
        where TShape : unmanaged, IInternalShape<TXD>
        where TIndices : unmanaged, IInternalIndices<TXD>
        where TXD : IXD
    {
        TShape Shape(ArrayView1D<TShape, Stride1D.Dense> shapes);

        long ProjectIndex(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte shapeIdx, long index);
    }

    internal readonly struct InternalTensor<TShape, TIndices, TXD>(byte shapeIdx, long offset) : IInternalTensor<TShape, TIndices, TXD>
        where TShape : unmanaged, IInternalShape<TXD>
        where TIndices : unmanaged, IInternalIndices<TXD>
        where TXD : IXD
    {
        public readonly byte ShapeIdx = shapeIdx;
        public readonly long Offset = offset;
        public readonly TShape Shape(ArrayView1D<TShape, Stride1D.Dense> shapes)
            => shapes[ShapeIdx];

        public static long ProjectIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TShape from, TShape to, long indexFrom)
        {
            TIndices indicesFrom = default;
            for (int d = from.Rank - 1; d >= 0; d--)
            {
                int size = dimensions[from[d]].Size;
                indicesFrom[d] = (int)(indexFrom % size);
                indexFrom /= size;
            }

            long indexTo = 0;
            for (int d = 0; d < to.Rank; d++)
            {
                byte dimIdx = to[d];
                indexTo *= dimensions[dimIdx].Size;
                indexTo += indicesFrom[from.IndexOf(dimIdx)];
            }

            return indexTo;
        }

        public long ProjectIndex(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte shapeToIdx, long indexFrom)
            => ProjectIndex(dimensions, shapes[ShapeIdx], shapes[shapeToIdx], indexFrom);
    }
}

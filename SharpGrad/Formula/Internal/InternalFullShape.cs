using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Formula.Internal
{
    internal struct InternalFullShape<TShape, TIndices, TXD>
        where TShape : unmanaged, IInternalShape<TXD>
        where TIndices : unmanaged, IInternalStaticArray<int, TXD>
        where TXD : IXD
    {
        public static TIndices GetSizes(TShape shape, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions)
        {
            TIndices sizes = default;
            for (int d = 0; d < shape.Rank; d++)
            {
                sizes[d] = dimensions[shape[d]].Size;
            }
            return sizes;
        }

        public readonly BIndex<byte> Idx;
        public readonly TShape Shape;
        public TIndices Sizes;

        public InternalFullShape(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, BIndex<byte> shapeIdx)
        {
            Idx = shapeIdx;
            if (shapeIdx.IsEmpty)
            {
                Shape = default;
                Sizes = default;
                Sizes[0] = 1;
            }
            else
            {
                Shape = shapes[shapeIdx];
                Sizes = GetSizes(Shape, dimensions);
            }
        }

        public static TIndices GetIndices(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TShape from, long indexFrom)
        {
            TIndices indices = default;
            for (int d = from.Rank - 1; d >= 0; d--)
            {
                int size = dimensions[from[d]].Size;
                indices[d] = (int)(indexFrom % size);
                indexFrom /= size;
            }
            return indices;
        }

        public readonly TIndices GetIndices(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, long indexFrom)
            => GetIndices(dimensions, Shape, indexFrom);

        public static long GetIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TShape shape, TIndices indices)
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

        public readonly long GetIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions)
            => GetIndex(dimensions, Shape, Sizes);

        public static TIndices GetIndicesFrom(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TShape from, TIndices indicesFrom, TShape to)
        {
            TIndices indicesTo = default;
            for (int d = 0; d < to.Rank; d++)
            {
                indicesTo[d] = indicesFrom[from.IndexOf(to[d])];
            }
            return indicesTo;
        }

        public readonly TIndices GetIndicesFrom(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TIndices indicesFrom, TShape to)
            => GetIndicesFrom(dimensions, Shape, indicesFrom, to);

        public static long ProjectIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TShape from, long indexFrom, TShape to)
        {
            TIndices indicesFrom = GetIndices(dimensions, from, indexFrom);
            TIndices indicesTo = GetIndicesFrom(dimensions, from, indicesFrom, to);
            return GetIndex(dimensions, to, indicesTo);
        }

        public readonly long ProjectIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, long indexFrom, TShape to)
            => ProjectIndex(dimensions, Shape, indexFrom, to);
    }

    internal readonly struct InternallFullShapeIndices<TShape, TIndices, TXD>(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte shapeIdx, TIndices indices)
        where TShape : unmanaged, IInternalShape<TXD>
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

        public readonly InternalFullShape<TShape, TIndices, TXD> Shape = new InternalFullShape<TShape, TIndices, TXD>(shapes, dimensions, shapeIdx);
        public readonly TIndices Indices = indices;

        public InternallFullShapeIndices(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte shapeIdx, long offset)
            : this(shapes, dimensions, shapeIdx, GetIndices(new InternalFullShape<TShape, TIndices, TXD>(shapes, dimensions, shapeIdx).Sizes, offset))
        { }
    }
}

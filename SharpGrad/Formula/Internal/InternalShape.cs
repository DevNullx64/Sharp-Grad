using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Formula.Internal
{
    internal struct InternalShape<TDimensions, TLengths, TXD>
        where TDimensions : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
        where TLengths : unmanaged, IInternalStaticArray<int, TXD>
        where TXD : IXD
    {
        public static TLengths GetSizes(TDimensions shape, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions)
        {
            TLengths sizes = default;
            for (int d = 0; d < shape.Count && shape[d] != -1; d++)
            {
                sizes[d] = dimensions[shape[d]].Size;
            }
            return sizes;
        }

        public readonly BIndex<byte> Idx;
        public readonly TDimensions Shape;
        public TLengths Sizes;

        public InternalShape(ArrayView1D<TDimensions, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, BIndex<byte> shapeIdx)
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

        public static TLengths GetIndices(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TDimensions from, long indexFrom)
        {
            TLengths indices = default;
            for (int d = from.Count - 1; d >= 0; d--)
            {
                BIndex<byte> iDim = from[d];
                if (iDim.IsEmpty)
                {
                    indices[d] = 0;
                }
                else
                {
                    int size = dimensions[iDim].Size;
                    indices[d] = (int)(indexFrom % size);
                    indexFrom /= size;
                }
            }
            return indices;
        }

        public readonly TLengths GetIndices(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, long indexFrom)
            => GetIndices(dimensions, Shape, indexFrom);

        public static long GetIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TDimensions shape, TLengths indices)
        {
            long index = 0;
            for (int d = 0; d < shape.Count; d++)
            {
                BIndex<byte> iDim = shape[d];
                if (!iDim.IsEmpty)
                {
                    index *= dimensions[iDim].Size;
                    index += indices[d];
                }
            }
            return index;
        }

        public readonly long GetIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions)
            => GetIndex(dimensions, Shape, Sizes);

        public static TLengths GetIndicesFrom(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TDimensions from, TLengths indicesFrom, TDimensions to)
        {
            TLengths indicesTo = default;
            for (int d = 0; d < to.Count; d++)
            {
                BIndex<byte> iDim = to[d];
                if (iDim.IsEmpty)
                {
                    indicesTo[d] = 0;
                }
                else
                {
                    indicesTo[d] = indicesFrom[from.IndexOf(iDim)];
                }
            }
            return indicesTo;
        }

        public readonly TLengths GetIndicesFrom(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TLengths indicesFrom, TDimensions to)
            => GetIndicesFrom(dimensions, Shape, indicesFrom, to);

        public static long ProjectIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TDimensions from, long indexFrom, TDimensions to)
        {
            TLengths indicesFrom = GetIndices(dimensions, from, indexFrom);
            TLengths indicesTo = GetIndicesFrom(dimensions, from, indicesFrom, to);
            return GetIndex(dimensions, to, indicesTo);
        }

        public readonly long ProjectIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, long indexFrom, TDimensions to)
            => ProjectIndex(dimensions, Shape, indexFrom, to);
    }
}

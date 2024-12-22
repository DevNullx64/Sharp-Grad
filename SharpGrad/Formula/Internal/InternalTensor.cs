using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Formula.Internal
{
    internal interface IInternalTensor<TShape, TIndices, TXD>
        where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
        where TIndices : unmanaged, IInternalStaticArray<int, TXD>
        where TXD : IXD
    {
        SourceOfOperand Source { get; }
        long ProjectIndex(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte shapeIdx, long index);
    }

    internal readonly struct InternalTensor<TShape, TIndices, TXD>(SourceOfOperand source, byte shapeIdx, long offset) : IInternalTensor<TShape, TIndices, TXD>
        where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
        where TIndices : unmanaged, IInternalStaticArray<int, TXD>
        where TXD : IXD
    {
        public readonly BIndex<byte> ShapeIdx = shapeIdx;
        public readonly long Offset = offset;

        public readonly SourceOfOperand Source { get; } = source;

        public static long ProjectIndex(ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, TShape from, TShape to, long indexFrom)
        {
            TIndices indicesFrom = default;
            for (int d = from.Count - 1; d >= 0; d--)
            {
                BIndex<byte> iDim = from[d];
                if (iDim.IsEmpty)
                {
                    indicesFrom[d] = 0;
                }
                else
                {
                    int size = dimensions[iDim].Size;
                    indicesFrom[d] = (int)(indexFrom % size);
                    indexFrom /= size;
                }
            }

            BIndex<byte> dimIdx = to[0];
            long indexTo = dimIdx.IsEmpty ? 0 : indicesFrom[from.IndexOf(dimIdx)];
            for (int d = 1; d < to.Count; d++)
            {
                if (!dimIdx.IsEmpty)
                    indexTo *= dimensions[dimIdx].Size;

                dimIdx = to[d];

                if (!dimIdx.IsEmpty)
                    indexTo += indicesFrom[from.IndexOf(dimIdx)];
            }

            return indexTo;
        }
        public static long ProjectIndex(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte fromIdx, byte toIdx, long indexFrom)
            => ProjectIndex(dimensions, shapes[fromIdx], shapes[toIdx], indexFrom);


        public long ProjectIndex(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, byte shapeToIdx, long indexFrom)
            => ProjectIndex(dimensions, shapes[ShapeIdx], shapes[shapeToIdx], indexFrom);
    }
}

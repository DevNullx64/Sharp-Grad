using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Formula.Internal
{
    public readonly struct AcceleratorShape<TShape, TIndices, TXD>
        where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
        where TIndices : unmanaged, IInternalStaticArray<int, TXD>
        where TXD : IXD
    {
        public readonly BIndex<byte> Idx;
        public readonly TShape DimsIndex;
        public readonly TIndices DimsSize;

        public AcceleratorShape(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, BIndex<byte> idx)
        {
            Idx = idx;
            DimsIndex = shapes[idx];
            DimsSize = default;
            for (int d = 0; d < DimsIndex.Count && DimsIndex[d] != -1; d++)
            {
                DimsSize[d] = dimensions[DimsIndex[d]].Size;
            }
        }

        public readonly AcceleratorShapeIndices<TShape, TIndices, TXD> ComputeIndices(long flatIndex)
        {
            TIndices indices = default;
            for (int d = DimsIndex.Count - 1; d >= 0; d--)
            {
                BIndex<byte> iDim = DimsIndex[d];
                if (iDim.IsEmpty)
                {
                    indices[d] = 0;
                }
                else
                {
                    int size = DimsSize[d];
                    indices[d] = (int)(flatIndex % size);
                    flatIndex /= size;
                }
            }
            return new AcceleratorShapeIndices<TShape, TIndices, TXD>(this, indices);
        }

        public readonly long ComputeFlatIndex(AcceleratorShapeIndices<TShape, TIndices, TXD> indices)
        {
            if(indices.Shape.Idx == Idx)
            {
                return indices.ComputeFlatIndex();
            }

            long flatIndex = 0;
            for (int d = 0; d < DimsIndex.Count; d++)
            {
                BIndex<byte> iDim = DimsIndex[d];
                if (!iDim.IsEmpty)
                {
                    int iFromIndices = indices.Shape.DimsIndex.IndexOf(iDim);
                    flatIndex *= indices.Shape.DimsSize[iFromIndices];
                    flatIndex += indices.Indices[iFromIndices];
                }
            }
            return flatIndex;
        }
    }

    public struct AcceleratorShapeIndices<TShape, TIndices, TXD>
        where TShape : unmanaged, IInternalStaticArray<BIndex<byte>, TXD>
        where TIndices : unmanaged, IInternalStaticArray<int, TXD>
        where TXD : IXD
    {
        public AcceleratorShapeIndices(AcceleratorShape<TShape, TIndices, TXD> acceleratorShape, TIndices indices = default)
        {
            Shape = acceleratorShape;
            Indices = indices;
        }

        public AcceleratorShapeIndices(ArrayView1D<TShape, Stride1D.Dense> shapes, ArrayView1D<InternalDimension, Stride1D.Dense> dimensions, BIndex<byte> idx, TIndices indices = default)
            : this(new AcceleratorShape<TShape, TIndices, TXD>(shapes, dimensions, idx), indices)
        { }

        public readonly AcceleratorShape<TShape, TIndices, TXD> Shape;
        public TIndices Indices;

        internal long ComputeFlatIndex()
        {
            long flatIndex = 0;
            for (int d = 0; d < Shape.DimsIndex.Count; d++)
            {
                BIndex<byte> iDim = Shape.DimsIndex[d];
                if (!iDim.IsEmpty)
                {
                    flatIndex *= Shape.DimsSize[d];
                    flatIndex += Indices[d];
                }
            }
            return flatIndex;
        }
    }
}
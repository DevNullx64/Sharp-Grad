using System;
using System.Collections;
using System.Collections.Generic;

namespace SharpGrad
{
    public class ShapeIndex : IReadOnlyList<DimIndex>
    {
        public Shape Shape { get; }

        private readonly Index[] indices;
        public Index[] Indices => indices;

        public int Count => indices.Length;

        public DimIndex this[int index] => new(Shape[index], indices[index]);

        public ShapeIndex(Shape shape, params Index[] indices)
        {
            if (shape.Length != indices.Length)
                throw new ArgumentException($"The number of indices must match the number of dimensions. Expected {shape.Length}, got {indices.Length}.");

            for (int i = 0; i < indices.Length; i++)
            {

                if (indices[i].Value < -shape[i].Size || indices[i].Value >= shape[i].Size)
                    throw new ArgumentOutOfRangeException(nameof(indices), $"The index must be between -{shape[i].Size} and {shape[i].Size - 1}. Got {indices[i].Value}.");
            }
            Shape = shape;
            this.indices = indices;
        }

        public static int GetFlattenIndex(Shape shape, params Index[] indices)
        {
            if (indices.Length != shape.Length)
                throw new ArgumentException($"The number of indices must match the number of dimensions. Expected {shape.Length}, got {indices.Length}.");
            var index = shape[0].Size * indices[0].Value;
            for (int i = 1; i < indices.Length; i++)
            {
                index *= shape[i].Size * indices[i].Value;
            }
            return index;
        }

        public int GetFlattenIndex()
            => GetFlattenIndex(Shape, Indices);

        public DimIndex[] GetIndices(int flattenedIndex)
        {
            if (flattenedIndex < 0 || flattenedIndex >= Shape.Length)
                throw new ArgumentOutOfRangeException(nameof(flattenedIndex), $"The index must be between 0 and {Shape.Length - 1}. Got {flattenedIndex}.");
            DimIndex[] dimIndices = new DimIndex[Shape.Length];
            for (int i = (int)Shape.Length - 1; i >= 0; i--)
            {
                Dimension dim = Shape[i];
                dimIndices[i] = new DimIndex(dim, flattenedIndex % dim.Size);
                flattenedIndex /= dim.Size;
            }
            return dimIndices;
        }

        public DimIndex[] GetDimIndices()
        {
            DimIndex[] dimIndices = new DimIndex[Shape.Length];
            for (int i = 0; i < Shape.Length; i++)
            {
                dimIndices[i] = this[i];
            }
            return dimIndices;
        }

        public IEnumerator<DimIndex> GetEnumerator()
        {
            for (int i = 0; i < Count; i++)
            {
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();
    }
}
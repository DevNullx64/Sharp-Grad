using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SharpGrad
{
    public class Shape(params Dimension[] dimensions) : IShape
    {
        public static Shape Broadcast(params Shape[] shapes)
        {
            List<Dimension> dims = [.. shapes[0]];
            for (int i = 1; i < shapes.Length; i++)
            {
                foreach (var dim in shapes[i])
                {
                    if (!dims.Contains(dim))
                    {
                        dims.Add(dim);
                    }
                }
            }
            return new Shape(dims);
        }

        public Shape(IEnumerable<Dimension> dimensions) : this(dimensions.Distinct().ToArray()) { }

        public int Count => dimensions.Length;

        public long Length { get; } = dimensions.Aggregate(1, (acc, dim) => acc * dim.Size);

        public bool IsScalar => Length == 1;

        public Dimension this[Index index] => dimensions[index];
        public Shape this[params Range[] ranges]
        {
            get
            {
                List<Dimension> dims = [];
                foreach (var range in ranges)
                {
                    dims.AddRange(dimensions[range]);
                }
                return new Shape(dims);
            }
        }

        public Dimension this[int index] => dimensions[index];

        public bool Contains(Dimension item) => dimensions.Contains(item);

        public IEnumerator<Dimension> GetEnumerator()
            => dimensions.AsEnumerable().GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator()
            => dimensions.GetEnumerator();

        public override string ToString()
            => $"({string.Join(", ", dimensions.AsEnumerable())})";

        public bool IsProperSubsetOf(IEnumerable<Dimension> other)
        {
            bool found = false;
            foreach (var dim in other)
            {
                if (!dimensions.Contains(dim))
                {
                    if (found)
                    {
                        return false;
                    }
                    found = true;
                }
            }
            return found;
        }

        public bool IsProperSupersetOf(IEnumerable<Dimension> other)
        {
            bool found = false;
            foreach (var dim in dimensions)
            {
                if (!other.Contains(dim))
                {
                    if (found)
                    {
                        return false;
                    }
                    found = true;
                }
            }
            return found;
        }

        public bool IsSubsetOf(IEnumerable<Dimension> other)
        {
            foreach (var dim in dimensions)
            {
                if (!other.Contains(dim))
                {
                    return false;
                }
            }
            return true;
        }

        public bool IsSupersetOf(IEnumerable<Dimension> other)
        {
            foreach (var dim in other)
            {
                if (!dimensions.Contains(dim))
                {
                    return false;
                }
            }
            return true;
        }

        public bool Overlaps(IEnumerable<Dimension> other)
        {
            foreach (var dim in dimensions)
            {
                if (other.Contains(dim))
                {
                    return true;
                }
            }
            return false;
        }

        public bool SetEquals(IEnumerable<Dimension> other)
            => IsSubsetOf(other) && IsSupersetOf(other);

        public static int GetFlattenIndex(params DimIndex[] indices)
        {
            var indice = indices[0];
            var index = indice.Dimention.Size * indice.Index.Value;
            for (int i = 1; i < indices.Length; i++)
            {
                indice = indices[i];
                index += indice.Dimention.Size * indice.Index.GetOffset(indice.Dimention.Size);
            }
            return index;
        }

        private static int GetOffset(Dimension dimension, Index index)
        {
            int offset = index.GetOffset(dimension.Size);
            if (offset < 0 || offset >= dimension.Size)
                throw new ArgumentOutOfRangeException(nameof(index), $"The index must be between 0 and {dimension.Size - 1}. Got {offset}.");
            return offset;
        }

        public int GetFlattenIndex(params Index[] indices)
        {
            if(indices.Length != dimensions.Length)
                throw new ArgumentException($"The number of indices must be {dimensions.Length}. Got {indices.Length}.");

            var index = GetOffset(dimensions[0], indices[0]);
            for (int i = 1; i < indices.Length; i++)
            {
                index *= dimensions[i].Size;
                index += GetOffset(dimensions[i], indices[i]);
            }
            return index;
        }

        public DimIndex[] GetIndices(int flattenedIndex)
        {
            if (flattenedIndex < 0 || flattenedIndex >= Length)
                throw new ArgumentOutOfRangeException(nameof(flattenedIndex), $"The index must be between 0 and {Length - 1}. Got {flattenedIndex}.");

            DimIndex[] indices = new DimIndex[dimensions.Length];
            for (int i = dimensions.Length - 1; i >= 0; i--)
            {
                indices[i] = new DimIndex(dimensions[i], flattenedIndex % dimensions[i].Size);
                flattenedIndex /= dimensions[i].Size;
            }
            return indices;
        }

        public DimIndex[] GetDimIndices(int flattenedIndex)
        {
            if (flattenedIndex < 0 || flattenedIndex >= Length)
                throw new ArgumentOutOfRangeException(nameof(flattenedIndex), $"The index must be between 0 and {Length - 1}. Got {flattenedIndex}.");

            var indices = new DimIndex[dimensions.Length];
            for (int i = dimensions.Length - 1; i >= 0; i--)
            {
                Dimension dim = dimensions[i];
                indices[i] = new DimIndex(dim, flattenedIndex % dim.Size);
                flattenedIndex /= dim.Size;
            }
            return indices;
        }

        public bool Equals(IShape? other)
        {
            if (other is null)
                return false;
            if (ReferenceEquals(this, other))
                return true;

            return Count == other.Count && dimensions.All(dim => other.Contains(dim));
        }
    }
}
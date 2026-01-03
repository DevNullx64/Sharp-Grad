using SharpGrad.DifEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SharpGrad
{
    public readonly struct Shape(params Dimension[] dimensions) : IReadOnlyList<Dimension>, IEquatable<Shape>
    {
        private readonly Dimension[]? dimensions = dimensions;

        public int Rank => dimensions?.Length ?? 0;
        int IReadOnlyCollection<Dimension>.Count => Rank;

        public bool IsScalar => Rank == 0;
        public bool IsVector => Rank == 1;

        public Dimension this[int index] => dimensions?[index] ?? throw new IndexOutOfRangeException($"Index {index} is out of range for shape with rank {Rank}.");
        public Dimension this[Index index] => this[index.GetOffset(Rank)];
        public IEnumerator<Dimension> GetEnumerator()
        {
            if (dimensions is not null)
            {
                for (int i = 0; i < dimensions.Length; i++)
                {
                    yield return dimensions[i];
                }
            }
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public int Size
        {
            get
            {
                int size = 1;
                if(dimensions is not null)
                {
                    for(int i = 0; i < dimensions.Length; i++)
                    {
                        size *= dimensions[i].Size;
                    }
                }
                return size;
            }
        }

        public int GetLinearIndex(IReadOnlyList<int> indices)
        {
            if (Rank == 0 && indices.Count == 1)
                return 0;

            if (Rank != indices.Count)
            {
                throw new ArgumentException($"The shape size {Size} is not equal to the indices length {indices.Count}");
            }

            int index = 0;
            int stride = 1;
            for (int i = Rank - 1; i >= 0; i--)
            {
                checked
                {
                    index += indices[i] * stride;
                    stride *= dimensions![i].Size;
                }
            }
            return index;
        }

        private int[] GetLocalIndices(Dimdices indices)
        {
            if (indices.IsScalar)
            {
                return [0];
            }
            if (this == indices.Shape)
            {
                return [.. indices.Indices];
            }
            else
            {
                int[] localIndice = new int[Rank];
                for (int i = localIndice.Length - 1; i >= 0; i--)
                {
                    Dimension dim = this[i];
                    Index index = indices[dim];
                    int idx = index.Value;
                    if (index.IsFromEnd)
                    {
                        if (idx > dim.Size)
                        {
                            throw new IndexOutOfRangeException($"Index {idx} is out of range for dimension {dim.Size}");
                        }
                        localIndice[i] = dim.Size - idx;
                    }
                    else
                    {
                        if (idx >= dim.Size)
                        {
                            throw new IndexOutOfRangeException($"Index {idx} is out of range for dimension {dim.Size}");
                        }
                        localIndice[i] = idx;
                    }
                }
                return localIndice;
            }
        }

        public int GetLinearIndex(Dimdices dimdices)
        {
            int[] loaclIndices = GetLocalIndices(dimdices);
            return GetLinearIndex(loaclIndices);
        }
        #region Equality
        public bool Equals(Shape other)
        {
            if (dimensions is null)
                return other.Rank == 0;

            if(other.dimensions is null)
                return dimensions.Length == 0;

            if (dimensions.Length != other.dimensions.Length)
                return false;

            for (int i = 0; i < Rank; i++)
            {
                if (dimensions[i] != other.dimensions[i])
                    return false;
            }
            return true;
        }
        public override bool Equals(object? obj) => obj is Shape other && Equals(other);

        public static bool operator ==(Shape left, Shape right) => left.Equals(right);
        public static bool operator !=(Shape left, Shape right) => !left.Equals(right);

        public override int GetHashCode()
        {
            int hash = typeof(Shape).GetHashCode();
            for (int i = 0; i < Rank; i++)
            {
                hash = hash * 31 + dimensions[i].GetHashCode();
            }
            return hash;
        }
        #endregion

        public override string ToString() => IsScalar ? "[]" : "[" + string.Join(", ", dimensions!.Select(d => d.Name)) + "]";

        #region Implicit cast operators
        public static implicit operator Shape(Dimension[] dimensions)
            => new(dimensions);
        public static implicit operator Shape(Dimension dimension)
            => new(dimension);
        public static implicit operator Shape((Dimension, Dimension) dimensions)
            => new(dimensions.Item1, dimensions.Item2);
        public static implicit operator Shape((Dimension, Dimension, Dimension) dimensions)
            => new(dimensions.Item1, dimensions.Item2, dimensions.Item3);
        public static implicit operator Shape((Dimension, Dimension, Dimension, Dimension) dimensions)
            => new(dimensions.Item1, dimensions.Item2, dimensions.Item3, dimensions.Item4);
        #endregion
    }
}

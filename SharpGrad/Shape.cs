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

        public int GetLinearIndex(params int[] indices)
        {
            if (Rank == 0 && indices.Length == 1)
                return 0;
            if (Rank != indices.Length)
            {
                throw new ArgumentException($"The shape size {Size} is not equal to the indices length {indices.Length}");
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

        #region Equality
        public bool Equals(Shape other)
        {
            if (Rank != other.Rank)
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

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SharpGrad
{
    public readonly struct Shape(params Dim[] dims) : IReadOnlyList<Dim>, IEquatable<Shape>
    { 
        private readonly Dim[] dims = dims;

        public Dim this[int index] => dims[index];
        public int Count => dims.Length;

        public int Size => dims.Aggregate(1, (a, b) => a * b);

        public int GetFlattenedIndex(int indice, params int[] indices)
        {
            if (indices.Length + 1 != Count)
                throw new ArgumentException($"Expected {Count} indices, got {indices.Length + 1}");
            return FlattenFrom(this, indice, indices);
        }
        public int[] GetIndices(int flattenedIndex)
        {
            if (flattenedIndex < 0 || flattenedIndex >= dims.Aggregate(1, (a, b) => a * b))
                throw new ArgumentOutOfRangeException(nameof(flattenedIndex));
            return IndicesFrom(this, flattenedIndex);
        }

        public IEnumerator<Dim> GetEnumerator() => ((IEnumerable<Dim>)dims).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => dims.GetEnumerator();

        public static int FlattenFrom(Shape Shape, int indice, params int[] indices)
        {
            if (indices.Length + 1 != Shape.Count)
                throw new ArgumentException($"Expected {Shape.Count} indices, got {indices.Length + 1}");

            int flattenedIndex = indice;
            for (int i = 0; i < indices.Length; i++)
            {
                flattenedIndex *= Shape[i];
                flattenedIndex += indices[i];
            }

            return flattenedIndex;
        }
        public static int[] IndicesFrom(Shape shape, int flattenedIndex)
        {
            int[] indices = new int[shape.Count];
            for (int i = shape.Count - 1; i >= 0; i--)
            {
                indices[i] = flattenedIndex % shape[i];
                flattenedIndex /= shape[i];
            }

            return indices;
        }

        public bool Equals(Shape other) => dims.SequenceEqual(other.dims);

        public override bool Equals(object? obj) => obj is Shape shape && Equals(shape);
        public static bool operator ==(Shape left, Shape right) => left.Equals(right);
        public static bool operator !=(Shape left, Shape right) => !left.Equals(right);

        public override int GetHashCode() => HashCode.Combine(dims);

        public override string ToString() => $"[{string.Join(", ", dims)}]";
    }
}

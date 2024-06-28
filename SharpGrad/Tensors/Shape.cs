using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SharpGrad
{
    /// <summary>
    /// Represents the shape of a tensor.
    /// </summary>
    /// <param name="dims">The dimensions of the tensor.</param>
    public readonly struct Shape(IEnumerable<Dim> dims) : IShape, IEquatable<Shape>
    {
        public Shape(params Dim[] dims) : this((IEnumerable<Dim>)dims) { }

        /// <summary>
        /// An empty shape.
        /// </summary>
        public static readonly Shape Empty = new();

        /// <summary>
        /// The dimensions of the tensor.
        /// </summary>
        private readonly Dim[] dims = dims.ToArray();

        /// <summary>
        /// Gets the dimension at the specified index.
        /// </summary>
        public Dim this[Index index] => dims[index];
        Dim IReadOnlyList<Dim>.this[int index] => dims[index];

        /// <summary>
        /// Gets the dimensions in the specified range.
        /// </summary>
        /// <param name="range">The range of dimensions to get.</param>
        /// <returns>The dimensions in the specified range.</returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public Dim[] this[Range range]
        {
            get
            {
                // compute the range
                int start = range.Start.IsFromEnd ? Count - range.Start.Value : range.Start.Value;
                int end = range.End.IsFromEnd ? Count - range.End.Value : range.End.Value;

                if(start < 0 || start >= Count)
                    throw new ArgumentOutOfRangeException(nameof(range), start, "Start index out of range.");
                if(end < 0 || end >= Count)
                    throw new ArgumentOutOfRangeException(nameof(range), end, "End index out of range.");
                if(start > end)
                    throw new ArgumentOutOfRangeException(nameof(range), "Start index is greater than end index.");

                Dim[] result = new Dim[end - start];
                for (int i = start; i < end; i++)
                    result[i - start] = dims[i];

                return result;
            }
        }

        /// <summary>
        /// Gets the number of dimensions in the tensor.
        /// </summary>
        public int Count => dims.Length;

        private readonly long length = dims.Aggregate(1L, (a, b) => a * b);
        /// <summary>
        /// Gets the total number of elements in the tensor.
        /// </summary>
        public long Length => length;


        /// <summary>
        /// Gets a value indicating whether the tensor is a scalar.
        /// </summary>
        public bool IsScalar { get => Length == 1; }

        /// <summary>
        /// Gets the flattened index from the specified indices.
        /// </summary>
        /// <param name="indices">The indices to flatten.</param>
        /// <returns>The flattened index.</returns>
        public int GetFlattenIndex(params Index[] indices) => FlattenFrom(this, indices);

        /// <summary>
        /// Gets the indices from the specified flattened index.
        /// </summary>
        /// <param name="flattenedIndex">The flattened index to get the indices from.</param>
        /// <returns>The indices from the specified flattened index.</returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public Index[] GetIndices(int flattenedIndex)
        {
            if (flattenedIndex < 0 || flattenedIndex >= Length)
                throw new ArgumentOutOfRangeException(nameof(flattenedIndex));
            return IndicesFrom(this, flattenedIndex);
        }

        /// <inheritdoc/>
        public IEnumerator<Dim> GetEnumerator() => ((IEnumerable<Dim>)dims).GetEnumerator();
        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() => dims.GetEnumerator();

        /// <summary>
        /// Flattens the specified indices.
        /// </summary>
        /// <param name="Shape">The shape of the tensor.</param>
        /// <param name="indices">The indices to flatten.</param>
        /// <returns>The flattened index.</returns>
        /// <exception cref="ArgumentException"></exception>
        public int FlattenFrom(params Index[] indices)
        {
            if (indices.Length != dims.Length)
                throw new ArgumentException($"Expected {Count} indices, got {indices.Length}");

            int flattenedIndex = 0;

            for (int i = 0; i < dims.Length; i++)
            {
                if (indices[i].Value < 0 || indices[i].Value >= dims[i])
                    throw new ArgumentOutOfRangeException(nameof(indices), indices[i].Value, $"Index out of range for dimension {i}. {indices[0].Value} is not in the range [0, {dims[i]})");

                flattenedIndex *= dims[i];
                flattenedIndex += indices[i].IsFromEnd
                    ? dims[i] - indices[i].Value
                    : indices[i].Value;
            }

            return flattenedIndex;
        }

        /// <summary>
        /// Gets the indices from the specified flattened index.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="flattenedIndex">The flattened index to get the indices from.</param>
        /// <returns>The indices from the specified flattened index.</returns>
        public static Index[] IndicesFrom(Shape shape, int flattenedIndex)
        {
            Index[] indices = new Index[shape.Count];
            for (int i = shape.Count - 1; i >= 0; i--)
            {
                indices[i] = flattenedIndex % shape[i];
                flattenedIndex /= shape[i];
            }

            return indices;
        }

        /// <inheritdoc/>
        public bool Equals(Shape other) => dims.SequenceEqual(other.dims);
        /// <inheritdoc/>
        public override bool Equals(object? obj) => obj is Shape shape && Equals(shape);
        /// <inheritdoc/>
        public static bool operator ==(Shape left, Shape right) => left.Equals(right);
        /// <inheritdoc/>
        public static bool operator !=(Shape left, Shape right) => !left.Equals(right);

        /// <inheritdoc/>
        public override int GetHashCode() => HashCode.Combine(dims);

        /// <inheritdoc/>
        public override string ToString() => $"[{string.Join(", ", dims)}]";

        /// <summary>
        /// Implicitly converts an array of dimensions to a shape.
        /// </summary>
        /// <param name="dims">The dimensions of the tensor.</param>
        /// <returns>The shape of the tensor.</returns>
        public static implicit operator Shape(Dim[] dims) => new(dims);

        /// <summary>
        /// Implicitly converts an array of integers to a shape.
        /// </summary>
        /// <param name="dims">The dimensions of the tensor.</param>
        /// <returns>The shape of the tensor.</returns>
        public static implicit operator Shape(int[] dims) => new(dims.Select(x => (Dim)x).ToArray());
    }
}

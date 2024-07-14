using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace SharpGrad
{
    /// <summary>
    /// Represents the shape of a tensor.
    /// </summary>
    /// <param name="dims">The dimensions of the tensor.</param>
    public readonly struct Shape(IEnumerable<int> dims) : IShape, IEquatable<Shape>
    {
        public Shape(params int[] ints) : this(ints.AsEnumerable()) { }

        /// <summary>
        /// An empty shape.
        /// </summary>
        public static readonly Shape Empty = new();

        /// <summary>
        /// The dimensions of the tensor.
        /// </summary>
        private readonly int[] dims = dims.ToArray();

        /// <summary>
        /// Gets the dimension at the specified index.
        /// </summary>
        public int this[Index index] => dims[index];
        int IReadOnlyList<int>.this[int index] => dims[index];

        /// <summary>
        /// Gets the dimensions in the specified range.
        /// </summary>
        /// <param name="range">The range of dimensions to get.</param>
        /// <returns>The dimensions in the specified range.</returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public int[] this[Range range]
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

                int[] result = new int[end - start];
                for (int i = start; i < end; i++)
                    result[i - start] = dims[i];

                return result;
            }
        }

        /// <summary>
        /// Gets the number of dimensions in the tensor.
        /// </summary>
        public int Count => dims.Length;

        private readonly long length = GetLength(dims);
        /// <summary>
        /// Gets the total number of elements in the tensor.
        /// </summary>
        public long Length => length;


        /// <summary>
        /// Gets a value indicating whether the tensor is a scalar.
        /// </summary>
        public bool IsScalar { get => Length == 1; }

        public Shape SetDim(Index dim, int size)
        {
            int[] newDims = (int[])dims.Clone();
            newDims[dim.IsFromEnd ? Count - dim.Value : dim.Value] = size;
            return new Shape(newDims);
        }
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
        public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)dims).GetEnumerator();
        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() => dims.GetEnumerator();

        /// <summary>
        /// Flattens the specified indices.
        /// </summary>
        /// <param name="Shape">The shape of the tensor.</param>
        /// <param name="indices">The indices to flatten.</param>
        /// <returns>The flattened index.</returns>
        /// <exception cref="ArgumentException"></exception>
        public int GetFlattenIndex(params Index[] indices)
        {
            // check if the number of indices is equal to the number of dimensions
            if (indices.Length != dims.Length)
            {
                // check if the last dimension is 1 and the number of indices is equal to the number of dimensions - 1
                if (dims[^1] == 1 && indices.Length == dims.Length - 1)
                {
                    List<Index> newIndices = new(indices) { Index.FromStart(0) };
                    indices = [.. newIndices];
                }
                else
                    throw new ArgumentException($"Expected {dims.Length} indices, got {indices.Length}");
            }

            // convert the indices to integers
            int[] intsIndices = new int[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                // check if the index is from the end and convert it to an integer
                intsIndices[i] = indices[i].IsFromEnd ? dims[i] - indices[i].Value : indices[i].Value;
                if (intsIndices[i] < 0 || intsIndices[i] >= dims[i])
                    throw new ArgumentOutOfRangeException(nameof(indices), i, "Index out of range.");
            }

            return GetFlattenIndices(dims, intsIndices);
        }


        /// <summary>
        /// Flattens the specified <paramref name="indices"/> within the <paramref name="shape"/>.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="indices">The indices to flatten.</param>
        /// <returns>The flattened index.</returns>
        /// <remarks>This funtion considers the indices as column-major.</remarks>
        /// </remarks>
        public static int GetFlattenIndices(int[] shape, params int[] indices)
        {
            int r = 0;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                r *= shape[i];
                r += indices[i];
            }
            return r;
        }

        public static int GetFlattenIndices(ArrayView1D<int, Stride1D.Dense> shape, params int[] indices)
        {
            int r = 0;
            for (int i = shape.IntLength - 1; i >= 0; i--)
            {
                r *= shape[i];
                r += indices[i];
            }
            return r;
        }

        /// <summary>
        /// Gets the indices within the <paramref name="shape"/> from the specified <paramref name="flattenedIndex"/>.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="flattenedIndex">The flattened index to get the indices from.</param>
        /// <returns>The indices from the specified flattened index.</returns>
        /// <remarks>This funtion considers the indices as column-major.</remarks>
        public static int[] IndicesFrom(int[] shape, int flattenedIndex)
        {
            int[] results = new int[shape.Length];

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                results[i] = flattenedIndex % shape[i];
                flattenedIndex /= shape[i];
            }

            return results;
        }

        /// <summary>
        /// Adds the specified <paramref name="value"/> to the <paramref name="index"/> of the <paramref name="shape"/> at the specified <paramref name="dim"/>.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="index">Flattened index to add the value to.</param>
        /// <param name="dim">The dimension to add the value to.</param>
        /// <param name="value">The value to add.</param>
        /// <returns>The flattened index after adding the value.</returns>
        public static int Add(ArrayView1D<int, Stride1D.Dense> shape, int index, int dim, int value)
        {
            Debug.Assert(dim < shape.IntLength, $"Dimension {dim} is out of range.");
            Debug.Assert(shape[dim] > 0, $"Dimension {dim} is zero.");

            int size = shape[0];
            for (int i = 1; i < dim; i++)
                size *= shape[i];
            return index + value * size;
        }

        /// <summary>
        /// Gets the indices from the specified flattened index.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <param name="flattenedIndex">The flattened index to get the indices from.</param>
        /// <returns>The indices from the specified flattened index.</returns>
        public static int[] IndicesFrom(ArrayView1D<int, Stride1D.Dense> shape, int flattenedIndex)
        {
            int[] results = new int[shape.Length];

            for (int i = shape.IntLength - 1; i >= 0; i--)
            {
                results[i] = flattenedIndex % shape[i];
                flattenedIndex /= shape[i];
            }

            return results;
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

        internal Shape Reduce(Index dim)
        {
            var dims = (int[])this.dims.Clone();
            dims[dim] = 1;
            return new Shape(dims);
        }

        internal static int GetLength(IEnumerable<int> sourceShape)
        {
            int length = 1;
            foreach(int dim in sourceShape)
                length *= dim;
            return length;
        }

        /// <summary>
        /// Implicitly converts an array of integers to a shape.
        /// </summary>
        /// <param name="dims">The dimensions of the tensor.</param>
        /// <returns>The shape of the tensor.</returns>
        public static implicit operator Shape(int[] dims) => new(dims);
        public static explicit operator int[](Shape shape) => shape.dims;
    }
}

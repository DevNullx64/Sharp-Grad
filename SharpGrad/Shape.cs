using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    public readonly struct Shape : IReadOnlyList<Dimension>, IEquatable<Shape>
    {
        internal readonly Dimension[]? dimensions;
        internal readonly int[]? Strides;

        /// <summary>
        /// Creates a new Shape from the given dimensions.
        /// </summary>
        /// <param name="dims">The dimensions of the shape.</param>
        /// <remarks>
        /// Scalar dimensions (dimensions with size 1) are ignored.
        /// If all dimensions are scalar, the shape is considered scalar (rank 0).
        /// </remarks>
        public Shape(IEnumerable<Dimension> dims)
        {
            if (dims is not null)
            {
                Dimension[] dimsArray = new Dimension[dims.Count()];
                int iDimsArray = 0;
                foreach (Dimension d in dims)
                {
                    if (!d.IsScalar)
                    {
                        dimsArray[iDimsArray++] = d;
                    }
                }

                if (iDimsArray > 0)
                {
                    if(iDimsArray != dimsArray.Length)
                    {
                        dimsArray = dimsArray[..iDimsArray];
                    }
                    dimensions = dimsArray;
                    Strides = new int[dimensions.Length];
                    int stride = 1;
                    for (int i = dimensions.Length - 1; i >= 0; i--)
                    {
                        Strides[i] = stride;
                        stride *= dimensions[i].Size;
                    }
                    return;
                }
            }
            dimensions = null;
            Strides = null;
        }

        /// <summary>
        /// Creates a new Shape from the given dimensions.
        /// </summary>
        /// <param name="dims">The dimensions of the shape.</param>
        /// <remarks>
        /// Scalar dimensions (dimensions with size 1) are ignored.
        /// If all dimensions are scalar, the shape is considered scalar (rank 0).
        /// </remarks>
        public Shape(params Dimension[] dims)
            : this((IEnumerable<Dimension>)dims) { }

        public int Rank
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => dimensions is null ? 0 : dimensions.Length;
        }
        int IReadOnlyCollection<Dimension>.Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Rank;
        }

        public bool IsScalar
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Rank == 0;
        }
        public bool IsVector
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Rank == 1;
        }

        public Dimension this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => dimensions?[index] ?? throw new IndexOutOfRangeException($"Index {index} is out of range for scalar shape.");
        }
        public Dimension this[Index index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => this[index.GetOffset(Rank)];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetStride(int i) => Strides?[i] ?? throw new IndexOutOfRangeException($"Index {i} is out of range for scalar shape.");

        public int IndexOf(Dimension dimension)
        {
            if (dimensions is null)
                return -1;
            return Array.IndexOf(dimensions, dimension);
        }

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

        /// <summary>
        /// Gets the total size of the shape (the product of all dimension sizes).
        /// </summary>
        public int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (Strides?[0] ?? 1) * (dimensions?[0].Size ?? 1);
        }

        /// <summary>
        /// Broadcast two shapes together.
        /// </summary>
        /// <param name="a">The first shape.</param>
        /// <param name="b">The second shape.</param>
        /// <returns>
        /// The broadcasted shape.
        /// </returns>
        /// <remarks>
        /// The broadcasted shape contains all unique dimensions from both shapes.
        /// If one of the shapes is scalar, the other shape is returned.
        /// </remarks>
        public static Shape Broadcast(Shape a, Shape b)
        {
            if (a.IsScalar) return b;
            if (b.IsScalar) return a;
            List<Dimension> mergedDimensions = [.. a];
            foreach (var dim in b)
            {
                if (!mergedDimensions.Contains(dim))
                {
                    mergedDimensions.Add(dim);
                }
            }
            return new([.. mergedDimensions]);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Shape Broadcast(Shape other)
            => Broadcast(this, other);

        public static Shape Remove(Shape shape, params Dimension[] dimsToRemove)
        {
            int reducedDim = dimsToRemove.Where(d => !d.IsScalar).Count();
            List<Dimension> resultDims = new(shape.Rank);
            for(int i = 0; i < shape.Rank; i++)
            {
                Dimension dim = shape[i];
                if (!dimsToRemove.Contains(dim))
                {
                    resultDims.Add(dim);
                }
                else
                {
                    reducedDim--;
                }
            }
            if (reducedDim != 0)
            {
                throw new ArgumentException($"Some reduction dimensions were not found in the input shape. Original shape: {shape}, dimensions to remove: [{string.Join(", ", dimsToRemove)}]");
            }
            return new(resultDims);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Shape Remove(Shape shape, IEnumerable<Dimension> dimsToRemove)
            => Remove(shape, [.. dimsToRemove]);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Shape Remove(params Dimension[] dimsToRemove)
            => Remove(this, (IEnumerable<Dimension>)dimsToRemove);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Shape Remove(IEnumerable<Dimension> dimsToRemove)
            => Remove(this, dimsToRemove);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetLinearIndex(params int[] indices)
        {
            if(dimensions is null || Strides is null) // this.Rank == 0
            {
                if (indices.Length > 1)
                {
                    throw new ArgumentException($"The shape is scalar, but indices length is {indices.Length}.");
                }
                return 0;
            }
            int rank = dimensions.Length;
            if (rank != indices.Length)
            {
                throw new ArgumentException($"The shape is of rank {rank} which is not equal to the indices length {indices.Length}");
            }

            int idx = 0;
            for (int i = rank - 1; i >= 0; i--)
            {
                checked
                {
                    idx += indices[i] * Strides[i];
                }
            }
            return idx;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetLinearIndex(params Index[] indices)
        {
            if(dimensions is null) // this.Rank == 0
            {
                if (indices.Length != 0)
                {
                    throw new ArgumentException($"The shape is scalar, but indices length is {indices.Length}.");
                }
                return 0;
            }

            int rank = dimensions.Length;
            if (rank != indices.Length)
            {
                throw new ArgumentException($"The shape is of rank {rank} which is not equal to the indices length {indices.Length}");
            }

            int[] offsets = new int[rank];
            for (int i = rank - 1; i >= 0; i--)
            {
                offsets[i] = indices[i].GetOffset(dimensions[i].Size);
            }
            return GetLinearIndex(offsets);
        }

        /// <summary>
        /// Get the index in this shape from the index in another shape.
        /// </summary>
        /// <param name="fromIndex">The index in the other shape.</param>
        /// <param name="fromShape">The other shape. Must be broadcastable to this shape.</param>
        /// <param name="requireAllDimensions">
        /// When true, all dimensions of this shape must exist in <paramref name="fromShape"/>; otherwise an exception is thrown.
        /// When false, missing dimensions are treated as having index 0 (broadcasting).
        /// </param>
        /// <returns>The index in this shape.</returns>
        public static int GetLinearIndex(Shape toShape, int fromIndex, Shape fromShape, bool requireAllDimensions)
        {
            if (toShape.dimensions is null || toShape.Strides is null) // this.Rank == 0
            {
                return 0;
            }

            if (fromShape.dimensions is null || fromShape.Strides is null) // shape.Rank == 0
            {
                return requireAllDimensions
                    ? throw new ArgumentException("The provided shape is not broadcastable to this shape.")
                    : 0;
            }

            if (toShape == fromShape)
            {
                return fromIndex;
            }

            int resultIndex = 0;
            for (int itd = toShape.dimensions.Length - 1; itd >= 0; itd--)
            {
                Dimension dim = toShape.dimensions[itd];
                int iod = Array.IndexOf(fromShape.dimensions, dim);
                if (iod >= 0)
                {
                    int coord = fromIndex / fromShape.Strides[iod] % dim.Size;
                    resultIndex += coord * toShape.Strides[itd];
                }
                else if (requireAllDimensions)
                {
                    throw new ArgumentException("The provided shape is not broadcastable to this shape.");
                }
            }

            return resultIndex;
        }

        /// <summary>
        /// Get the index in this shape from the index in this shape.
        /// </summary>
        /// <param name="fromIndex">The index in the other shape.</param>
        /// <param name="fromShape">The other shape.</param>
        /// <param name="requireAllDimensions">When true, all dimensions of this shape must exist in <paramref name="fromShape"/>; otherwise an exception is thrown.
        /// When false, missing dimensions are treated as having index 0 (broadcasting).</param>
        /// <returns>The index in this shape.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetLinearIndex(int fromIndex, Shape fromShape, bool requireAllDimensions)
            => GetLinearIndex(this, fromIndex, fromShape, requireAllDimensions);

        public static void ThrowIfNotCompatible(Array array, Shape shape)
        {
            if (shape.IsScalar)
            {
                if (array.Rank != 1 || array.GetLength(0) != 1)
                {
                    throw new ArgumentException($"Scalar shape expects a rank-1 array of length 1. Got rank {array.Rank} length {array.GetLength(0)}");
                }
                return;
            }
            if (array.Rank != shape.Rank)
            {
                throw new ArgumentException($"The array rank {array.Rank} is not equal to the shape rank {shape.Rank}");
            }
            for (int d = 0; d < array.Rank; d++)
            {
                if (array.GetLength(d) != shape[d].Size)
                {
                    throw new ArgumentException($"The array dimension length {array.GetLength(d)} is not equal to the shape dimension size {shape[d].Size} at dimension {d}");
                }
            }
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
                hash = hash * 31 + dimensions![i].GetHashCode();
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
        public static implicit operator Shape((Dimension, Dimension, Dimension, Dimension, Dimension) dimensions)
            => new(dimensions.Item1, dimensions.Item2, dimensions.Item3, dimensions.Item4, dimensions.Item5);
        public static implicit operator Shape((Dimension, Dimension, Dimension, Dimension, Dimension, Dimension) dimensions)
            => new(dimensions.Item1, dimensions.Item2, dimensions.Item3, dimensions.Item4, dimensions.Item5, dimensions.Item6);
        public static implicit operator Shape((Dimension, Dimension, Dimension, Dimension, Dimension, Dimension, Dimension) dimensions)
            => new(dimensions.Item1, dimensions.Item2, dimensions.Item3, dimensions.Item4, dimensions.Item5, dimensions.Item6, dimensions.Item7);

        public static implicit operator Dimension[](Shape shape)
            => shape.dimensions is null
            ? []
            : shape.dimensions;
        #endregion
    }

    public static class ShapeExtensions
    {
        /// <summary>
        /// Get the index of the specified dimension in the shape.
        /// </summary>
        /// <param name="toShape">The shape to search.</param>
        /// <param name="dimension">The dimension to find.</param>
        /// <returns>The index of the dimension in the shape, or -1 if the dimension is not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetLinearIndex(this List<Dimension> toShape, int indexFrom, List<Dimension> shapeFrom, bool requireAllDimensions)
        {
            if (toShape.Count == 0)
            {
                return 0;
            }
            if (shapeFrom.Count == 0)
            {
                if (requireAllDimensions)
                {
                    throw new ArgumentException("The provided shape is not broadcastable to this shape.");
                }
                return 0;
            }

            // Précalculer les strides pour shapeFrom
            int[] stridesFrom = new int[shapeFrom.Count];
            int strideFrom = 1;
            for (int i = shapeFrom.Count - 1; i >= 0; i--)
            {
                stridesFrom[i] = strideFrom;
                strideFrom *= shapeFrom[i].Size;
            }

            int resultIndex = 0;
            int strideTo = 1;
            for (int itd = toShape.Count - 1; itd >= 0; itd--)
            {
                Dimension dim = toShape[itd];
                int iod = shapeFrom.IndexOf(dim);
                if (iod >= 0)
                {
                    int coord = indexFrom / stridesFrom[iod] % dim.Size;
                    resultIndex += coord * strideTo;
                }
                else if (requireAllDimensions)
                {
                    throw new ArgumentException("The provided shape is not broadcastable to this shape.");
                }
                strideTo *= dim.Size;
            }
            return resultIndex;
        }
    }
}

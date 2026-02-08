using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    public readonly struct Shape : IReadOnlyList<Dimension>, IEquatable<Shape>
    {
        private readonly Dimension[]? dimensions;
        private readonly int[]? Strides;

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
        public int Size => (Strides?[0] ?? 1) * (dimensions?[0].Size ?? 1);

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
        public Shape Remove(params Dimension[] dimsToRemove)
            => Remove(this, (IEnumerable<Dimension>)dimsToRemove);
        public static Shape Remove(Shape shape, IEnumerable<Dimension> dimsToRemove)
            => Remove(shape, [.. dimsToRemove]);
        public Shape Remove(IEnumerable<Dimension> dimsToRemove)
            => Remove(this, dimsToRemove);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetLinearIndex(params int[] indices)
        {
            int rank = Rank;
            if (rank == 0)
                return 0;

            if (rank != indices.Length)
            {
                throw new ArgumentException($"The shape is of rank {rank} which is not equal to the indices length {indices.Length}");
            }

            int idx = 0;
            for (int i = rank - 1; i >= 0; i--)
            {
                checked
                {
                    idx += indices[i] * Strides![i];
                }
            }
            return idx;
        }


        private int[] GetOffsetsFromIndices(Index[] indices)
        {
            int r = Rank;
            if (indices.Length != r)
            {
                throw new ArgumentException($"The shape size {Size} is not equal to the indices length {indices.Length}");
            }
            int[] offsets = new int[r];
            for (int i = r - 1; i >= 0; i--)
            {
                offsets[i] = indices[i].GetOffset(dimensions![i].Size);
            }
            return offsets;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetLinearIndex(params Index[] indices)
        {
            int[] offsets = GetOffsetsFromIndices(indices);
            return GetLinearIndex(offsets);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetLinearIndex(Dimdices dimdices)
        {
            int[] localIndices = GetLocalIndices(dimdices);
            return GetLinearIndex(localIndices);
        }

        /// <summary>
        /// Convert an index from one shape to another shape.
        /// </summary>
        /// <param name="at">The index in the 'from' shape.</param>
        /// <param name="from">The shape of the 'at' index.</param>
        /// <param name="to">The target shape to convert the index to.</param>
        public static int GetLinearIndex(int at, Shape from, Shape to)
        {
            if(from == to)
            {
                return at;
            }
            int toRank = to.Rank;
            if(toRank == 0)
            {
                return 0;
            }
            int fromRank = from.Rank;
            if (fromRank == 0)
            {
                throw new ArgumentException($"The '{nameof(from)}' shape is scalar, cannot convert index to non-scalar shape." );
            }
            int linearIndex = 0;
            int setted = 0;
            for (int indexFromShape = 0; indexFromShape < fromRank; indexFromShape++)
            {
                Dimension dim = from[indexFromShape];
                int i = at / from.GetStride(indexFromShape) % dim.Size;
                int indexToShape = to.IndexOf(dim);
                if (indexToShape >= 0)
                {
                    linearIndex += i * to.GetStride(indexToShape);
                    setted++;
                }
            }
            return linearIndex;
        }

        /// <summary>
        /// Get the index in this shape from the index in another shape.
        /// </summary>
        /// <param name="index">The index in the other shape.</param>
        /// <param name="shape">The other shape. Must be broadcastable to this shape.</param>
        /// <returns>The index in this shape.</returns>
        /// <remarks>
        /// This method doesn't use intermediate array allocations. It uses only integer arithmetic to compute the index based on precomputed strides.
        /// </remarks>
        public int GetLinearIndex(int index, Shape shape)
        {
            if (this == shape)
            {
                return index;
            }

            if (dimensions is null || Strides is null)
            {
                if (shape.dimensions is null)
                    return 0;
                else
                    throw new ArgumentException("The provided shape is not broadcastable to this shape.");
            }
            if(shape.dimensions is null || shape.Strides is null)
            {
                throw new ArgumentException("The provided shape is not broadcastable to this shape.");
            }

            int treated = 0;
            int resultIndex = 0;
            int thisLastDimIndex = dimensions.Length - 1;
            for (int odi = shape.dimensions.Length - 1; odi >= 0; odi--)
            {
                Dimension dim = shape.dimensions[odi];

                for (int tdi = thisLastDimIndex; tdi >= 0; tdi--)
                {
                    if (dimensions[tdi] == dim)
                    {
                        int dimSize = dim.Size;
                        int coord = index / shape.Strides[odi] % dimSize;
                        resultIndex += coord * Strides[tdi];
                        treated++;
                        break;
                    }
                }
            }

            if (treated == dimensions.Length)
            {
                return resultIndex;
            }
            throw new ArgumentException("The provided shape is not broadcastable to this shape.");
        }

        /// <summary>
        /// Get the local indices in this shape from the given dimdices.
        /// </summary>
        /// <param name="indices">The dimdices to convert.</param>
        /// <returns>The local indices in this shape.</returns>
        /// <remarks>
        /// If the dimdices shape is equal to this shape, the indices are returned as is.
        /// If the dimdices is scalar, the local indices are [0].
        /// Otherwise, the indices are converted to local indices based on the dimension sizes.
        /// </remarks>
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
                int rank = Rank;
                int[] localIndices = new int[rank];
                for (int i = rank - 1; i >= 0; i--)
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
                        localIndices[i] = dim.Size - idx;
                    }
                    else
                    {
                        if (idx >= dim.Size)
                        {
                            throw new IndexOutOfRangeException($"Index {idx} is out of range for dimension {dim.Size}");
                        }
                        localIndices[i] = idx;
                    }
                }
                return localIndices;
            }
        }

        public static int[] GetIndicesArray(int at, Shape from, Shape to)
        {
            int toRank = to.Rank;
            if (toRank == 0)
            {
                return [0];
            }

            int fromRank = from.Rank;
            if (toRank > fromRank)
            {
                throw new ArgumentException($"Impossible to get indices from shape with rank {fromRank} to shape with rank {toRank}.");
            }

            int[] indicesFrom = InternalGetIndicesArray(at, from);
            if (from == to)
            {
                return indicesFrom;
            }

            return InternalGetIndicesArray(indicesFrom, from, to);
        }

        public static int[] GetIndicesArray(int[] at, Shape from, Shape to)
        {
            int toRank = to.Rank;
            if (toRank == 0)
            {
                return [0];
            }
            int fromRank = from.Rank;
            if (toRank > fromRank)
            {
                throw new ArgumentException($"Impossible to get indices from shape with rank {fromRank} to shape with rank {toRank}.");
            }
            return InternalGetIndicesArray(at, from, to);
        }

        private static int[] InternalGetIndicesArray(int at, Shape from)
        {
            int fromRank = from.Rank;
            int[] indicesFrom = new int[fromRank];
            for (int i = fromRank - 1; i >= 0; i--)
            {
                Dimension dim = from[i];
                indicesFrom[i] = at / from.GetStride(i) % dim.Size;
            }
            return indicesFrom;
        }

        private static int[] InternalGetIndicesArray(int[] at, Shape from, Shape to)
        {
            if(from == to)
            {
                return at;
            }
            int toRank = to.Rank;
            int[] indicesTo = new int[toRank];
            int dimAssigned = 0;
            for (int i = 0; i < toRank; i--)
            {
                Dimension dim = to[i];
                int indexInFrom = from.IndexOf(dim);
                if (indexInFrom != -1)
                {
                    indicesTo[i] = at[indexInFrom];
                    dimAssigned++;
                }
            }
            if (dimAssigned != toRank)
            {
                throw new ArgumentException("The provided shape is not broadcastable to this shape.");
            }
            return indicesTo;
        }

        public static int[] GetIndicesArray(Dimdices indices, Shape toShape)
        {
            return GetIndicesArray([.. indices.Indices], indices.Shape, toShape);
        }

        public static void ThrowIfNotCompatible(Array array, Shape shape)
        {
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
        #endregion
    }
}

using SharpGrad.DifEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SharpGrad
{
    public readonly struct Shape : IReadOnlyList<Dimension>, IEquatable<Shape>
    {
        private readonly Dimension[]? dimensions;
        private readonly int[]? Strides;

        public Shape(params Dimension[] dims)
        {
            List<Dimension> dimensionsList = [.. dims.Where(d => !d.IsScalar)];
            if (dimensionsList.Count > 0)
            {
                dimensions = [.. dimensionsList];
                Strides = new int[dimensions.Length];

                int stride = 1;
                for (int i = dimensions.Length - 1; i >= 0; i--)
                {
                    Strides[i] = stride;
                    stride *= dimensions[i].Size;
                }
            } else
            {
                dimensions = null;
                Strides = null;
            }
        }

        public int Rank => dimensions?.Length ?? 0;
        int IReadOnlyCollection<Dimension>.Count => Rank;

        public bool IsScalar => Rank == 0;
        public bool IsVector => Rank == 1;

        public Dimension this[int index] => dimensions?[index] ?? throw new IndexOutOfRangeException($"Index {index} is out of range for scalar shape.");
        public Dimension this[Index index] => this[index.GetOffset(Rank)];

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

        public int Size => (Strides?[0] ?? 1) * (dimensions?[0].Size ?? 1);

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
                        int coord = (index / shape.Strides[odi]) % dimSize;
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

using System;
using System.Collections;
using System.Collections.Generic;

namespace SharpGrad
{
    /// <summary>
    /// Represents a collection of <see cref="Dimension"/> and index pairs.
    /// </summary>
    public class Dimdices
    {
        public static readonly Dimdices Scalar = new();
        public static readonly int[] ScalarIntIndices = [0];

        /// <summary>
        /// The shape of the <see cref="Dimdices"/>.
        /// </summary>
        public readonly Shape Shape;
        /// <summary>
        /// The indices of the <see cref="Dimdices"/>.
        /// </summary>
        public IReadOnlyList<int> Indices;

        /// <summary>
        /// Get if the <see cref="Dimdices"/> is a scalar.
        /// </summary>
        public bool IsScalar => Shape.IsScalar;

        /// <summary>
        /// Get the index of the specified <see cref="Dimension"/>.
        /// </summary>
        /// <param name="dim">The <see cref="Dimension"/> to get the index of.</param>
        /// <returns>The index of the specified <see cref="Dimension"/>.</returns>
        /// <exception cref="KeyNotFoundException">
        /// Thrown when the specified <see cref="Dimension"/> is not found in the shape.
        /// </exception>
        public int this[Dimension dim]
        {
            get
            {
                for (int i = 0; i < Shape.Rank; i++)
                {
                    if (Shape[i] == dim)
                    {
                        return Indices[i];
                    }
                }
                throw new KeyNotFoundException($"Dimension {dim} not found in the shape.");
            }
        }

        private Dimdices()
        {
            Shape = [];
            Indices = [];
        }

        // Convert Index[] to int[]
        private static int[] ToInts(Dimension[] shape, Index[] indices)
        {
            if(shape.Length != indices.Length)
            {
                throw new ArgumentException($"The shape size {shape.Length} is not equal to the indices length {indices.Length}");
            }

            int[] idxs = new int[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                if (indices[i].IsFromEnd)
                {
                    idxs[i] = shape[i].Size - indices[i].Value;
                }
                else
                {
                    idxs[i] = indices[i].Value;
                }
            }
            return idxs;
        }

        /// <summary>
        /// Create a new instance of <see cref="Dimdices"/>.
        /// </summary>
        /// <param name="shape">The shape of the <see cref="Dimdices"/>.</param>
        /// <param name="indices">The indices of the <see cref="Dimdices"/>.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when the shape have no dimensions or its dimensions count is not equal to the indices length.
        /// </exception>
        public Dimdices(Shape shape, int[] indices)
        {
            if (shape.Rank != indices.Length)
            {
                throw new ArgumentException($"The shape size {shape.Size} is not equal to the indices length {indices.Length}");
            }
            Shape = shape;
            Indices = indices;
        }

        /// <summary>
        /// Create a new instance of <see cref="Dimdices"/>.
        /// </summary>
        /// <param name="shape">The shape of the <see cref="Dimdices"/>.</param>
        /// <param name="indices">The indices of the <see cref="Dimdices"/>.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when the shape have no dimensions or its dimensions count is not equal to the indices length.
        /// </exception>
        public Dimdices(Dimension[] shape, Index[] indices)
            : this(shape, ToInts(shape, indices))
        { }


        public override string ToString() => $"[{string.Join(", ", Indices)}]";
    }

    /// <summary>
    /// Represents an indexer for <see cref="Dimdices"/>.
    /// Starts from the first index in each dimension and ends at the last index in each dimension.
    /// </summary>
    public class Dimdexer(Shape shape) : IEnumerable<Dimdices>, IEnumerator<Dimdices>
    {
        public readonly Shape Shape = shape;
        public int[] Indices = new int[shape.Rank];

        public Dimdices Current => new(Shape, Indices);

        object IEnumerator.Current => Current;

        public void Dispose()
        { }

        public IEnumerator<Dimdices> GetEnumerator()
            => this;
        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();

        private bool isFirst = true;
        private bool isFinished = false;
        public bool MoveNext()
        {
            if (isFinished)
            {
                return false;
            }

            if (isFirst)
            {
                Array.Fill(Indices, 0);
                isFirst = !isFirst;
                return true;
            }

            for (int i = Shape.Rank - 1; i >= 0; i--)
            {
                Indices[i]++;
                if (Indices[i] < Shape[i].Size)
                {
                    return true;
                }
                Indices[i] = 0;
            }
            isFinished = !isFinished;
            return false;
        }

        public void Reset()
        {
            isFirst = true;
            isFinished = false;
        }
    }
}
using System;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    /// <summary>
    /// Abstract base class for read-only shaped arrays, providing common properties.
    /// </summary>
    public abstract class ReadOnlyDataBuffer : IReadOnlyDataBuffer
    {
        internal Array? internalData;
        /// <summary>
        /// Gets the shape of the array.
        /// </summary>
        public Shape Shape
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get;
        }
        /// <summary>
        /// Gets the type of elements in the array.
        /// </summary>
        public Type ElementType
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get;
        }
        /// <summary>
        /// Gets a value indicating whether the array is initialized.
        /// </summary>
        public bool IsInitialized
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => internalData is not null;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ReadOnlyDataBuffer"/> class with the specified element type and shape.
        /// </summary>
        /// <param name="elementType">The type of elements in the array.</param>
        /// <param name="shape">The shape of the array.</param>
        protected ReadOnlyDataBuffer(Type elementType, Shape shape)
        {
            Shape = shape;
            ElementType = elementType;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ReadOnlyDataBuffer"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        protected ReadOnlyDataBuffer(Array data, Shape shape)
        {
            if (shape.IsScalar)
            {
                if (data.Rank != 1 || data.GetLength(0) != 1)
                {
                    throw new ArgumentException($"Scalar shape expects a rank-1 array of length 1. Got rank {data.Rank} length {data.GetLength(0)}.");
                }
            }
            else
            {
                if (data.Rank != shape.Rank)
                {
                    throw new ArgumentException($"Data rank {data.Rank} does not match shape rank {shape.Rank}.");
                }
                for (int i = 0; i < shape.Rank; i++)
                {
                    if (data.GetLength(i) != shape[i].Size)
                    {
                        throw new ArgumentException($"Data dimension {i} size {data.GetLength(i)} does not match shape dimension size {shape[i].Size}.");
                    }
                }
            }
            internalData = data;
            Shape = shape;
            ElementType = data.GetType().GetElementType()!;
        }
    }

    /// <summary>
    /// Abstract class for read-only shaped arrays of type T, with flat data access.
    /// </summary>
    /// <typeparam name="T">The element type of the array.</typeparam>
    public abstract class ReadOnlyShapedArray<T>(Array data, Shape shape) :
        ReadOnlyDataBuffer(data, shape),
        IReadOnlyDataBuffer<T>
    {
        private readonly T[] flatData = Unsafe.As<T[]>(data);

        /// <summary>
        /// Gets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        public T this[params int[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => flatData[Shape.GetLinearIndex(indices)];
        }
        /// <summary>
        /// Gets the element at the specified indices.
        /// </summary>
        /// <param name="indices">The indices of the element.</param>
        /// <returns>The element at the specified indices.</returns>
        public T this[params Index[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => flatData[Shape.GetLinearIndex(indices)];
        }
    }
}
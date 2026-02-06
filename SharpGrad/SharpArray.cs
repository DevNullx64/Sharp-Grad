using System;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{

    /// <summary>
    /// Abstract base class for mutable shaped arrays, providing initialization and freeing.
    /// </summary>
    public abstract class DataBuffer : ReadOnlyDataBuffer, IDataBuffer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer"/> class with the specified element type and shape.
        /// </summary>
        /// <param name="elementType">The type of elements in the array.</param>
        /// <param name="shape">The shape of the array.</param>
        protected DataBuffer(Type elementType, Shape shape)
            : base(elementType, shape) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(Array data, Shape shape)
            : base(data, shape) { }

        public abstract bool Initialize();

        public abstract void SetData(Array newData);

        protected virtual void NoLock_Free()
        {
            internalData = null;
        }
        /// <summary>
        /// Frees the array resources.
        /// </summary>
        public void Free()
        {
            lock (this) 
            {
                NoLock_Free();
            }
        }

        /// <summary>
        /// Creates a new shaped array with the specified element type and shape.
        /// </summary>
        /// <param name="elementType">The type of elements in the array.</param>
        /// <param name="shape">The shape of the array.</param>
        /// <returns>A new shaped array.</returns>
        public static DataBuffer Create(Type elementType, Shape shape)
        {
            return (DataBuffer)Activator.CreateInstance(
                typeof(DataBuffer<>).MakeGenericType(elementType),
                shape
            )!;
        }

        /// <summary>
        /// Creates a new shaped array with the specified element type and shape.
        /// </summary>
        /// <param name="elementType">The type of elements in the array.</param>
        /// <param name="shape">The shape of the array.</param>
        /// <returns>A new shaped array.</returns>
        public static DataBuffer Create(Array data, Shape shape)
        {
            Type elementType = data.GetType().GetElementType()
                ?? throw new ArgumentException($"The '{nameof(data)}' array has no element type.");
            return (DataBuffer)Activator.CreateInstance(
                typeof(DataBuffer<>).MakeGenericType(elementType),
                data, shape
            )!;
        }

        /// <summary>
        /// Creates a new shaped array of type T with the specified shape.
        /// </summary>
        /// <typeparam name="TType">The element type of the array.</typeparam>
        /// <param name="shape">The shape of the array.</param>
        /// <returns>A new shaped array of type T.</returns>
        public static DataBuffer<TType> Create<TType>(Shape shape)
            => new(shape);
    }

    /// <summary>
    /// Concrete class for mutable shaped arrays of type T, supporting lazy initialization and various constructors.
    /// </summary>
    /// <typeparam name="T">The element type of the array.</typeparam>
    public class DataBuffer<T> : DataBuffer, IDataBuffer<T>
    {
        internal T[]? flatData;

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified shape.
        /// </summary>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(Shape shape) : base(typeof(T), shape) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(Array data, Shape shape) : base(data, shape)
        {
            if(data.GetType().GetElementType() != typeof(T))
            {
                throw new ArgumentException($"The element type of the data {data.GetType().GetElementType()} does not match the specified type {typeof(T)}.");
            }
            Shape.ThrowIfNotCompatible(data, Shape);
            flatData = Unsafe.As<T[]>(data);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(T[] data, Shape shape) : base(data, shape)
        {
            flatData = Unsafe.As<T[]>(data);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(T[,] data, Shape shape) : base(data, shape)
        {
            flatData = Unsafe.As<T[]>(data);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(T[,,] data, Shape shape) : base(data, shape)
        {
            flatData = Unsafe.As<T[]>(data);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(T[,,,] data, Shape shape) : base(data, shape)
        {
            flatData = Unsafe.As<T[]>(data);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(T[,,,,] data, Shape shape) : base(data, shape)
        {
            flatData = Unsafe.As<T[]>(data);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(T[,,,,,] data, Shape shape) : base(data, shape)
        {
            flatData = Unsafe.As<T[]>(data);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DataBuffer{T}"/> class with the specified data and shape.
        /// </summary>
        /// <param name="data">The array data.</param>
        /// <param name="shape">The shape of the array.</param>
        public DataBuffer(T[,,,,,,] data, Shape shape) : base(data, shape)
        {
            flatData = Unsafe.As<T[]>(data);
        }

        public T this[params int[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => flatData![Shape.GetLinearIndex(indices)];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => flatData![Shape.GetLinearIndex(indices)] = value;
        }

        public T this[params Index[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => flatData![Shape.GetLinearIndex(indices)];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => flatData![Shape.GetLinearIndex(indices)] = value;
        }

        public T this[Dimdices indices]
        {
            get => flatData![Shape.GetLinearIndex(indices)];
            set => flatData![Shape.GetLinearIndex(indices)] = value;
        }

        protected bool NoLock_Initialize()
        {
            if (internalData is null)
            {
                internalData = Array.CreateInstance(typeof(T), Shape.Select(d => d.Size).ToArray());
                flatData = Unsafe.As<T[]>(internalData);
                return true;
            }
            return false;
        }
        public override bool Initialize()
        {
            lock (this)
            {
                return NoLock_Initialize();
            }
        }

        private void NoTypeCheck_SetData(Array newData)
        {
            lock (this)
            {
                int r = Shape.Rank;
                if (newData.Rank != r)
                {
                    throw new ArgumentException($"The rank of the new data {newData.Rank} does not match the existing data rank {r}.");
                }
                for (int i = 0; i < r; i++)
                {
                    if (Shape[i].Size != newData.GetLength(i))
                    {
                        throw new ArgumentException($"The dimension {i} of the new data {newData.GetLength(i)} does not match the existing data dimension {Shape[i].Size}.");
                    }
                }
                internalData = newData;
                flatData = Unsafe.As<T[]>(newData);
            }
        }
        public override void SetData(Array newData)
        {
            if (newData.GetType().GetElementType() != ElementType)
            {
                throw new ArgumentException($"The element type of the new data {newData.GetType().GetElementType()} does not match the existing data type {ElementType}.");
            }
            NoTypeCheck_SetData(newData);
        }

        public void SetData(T[] newData) => NoTypeCheck_SetData(newData);
        public void SetData(T[,] newData) => NoTypeCheck_SetData(newData);
        public void SetData(T[,,] newData) => NoTypeCheck_SetData(newData);
        public void SetData(T[,,,] newData) => NoTypeCheck_SetData(newData);
        public void SetData(T[,,,,] newData) => NoTypeCheck_SetData(newData);
        public void SetData(T[,,,,,] newData) => NoTypeCheck_SetData(newData);
        public void SetData(T[,,,,,,] newData) => NoTypeCheck_SetData(newData);

        public void Fill(T value)
        {
            lock (this)
            {
                NoLock_Initialize();
                Array.Fill(flatData!, value);
            }
        }

        /// <summary>
        /// Gets the initialized DataBuffer of type G from a untyped DataBuffer.
        /// </summary>
        /// <typeparam name="T">The type of the DataBuffer.</typeparam>
        /// <param name="buffer">The untyped DataBuffer.</param>
        /// <returns>The already initialized <see cref="DataBuffer{G}"/>.</returns>
        /// <remarks>
        /// Throws an InvalidOperationException if the buffer is not of the expected type or is not initialized.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the buffer is not of the expected type or is not initialized.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public DataBuffer<T> ThrowIfNotInitialized()
            => IsInitialized
            ? this
            : throw new InvalidOperationException("Buffer is not initialized.");

        /// <summary>
        /// Gets the initialized flat data array of type T from a untyped DataBuffer.
        /// </summary>
        /// <param name="buffer">The untyped DataBuffer.</param>
        /// <returns>The already initialized flat data array of type T.</returns>
        /// <remarks>
        /// Throws an InvalidOperationException if the buffer is not of the expected type or is not initialized.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal T[] GetInitializedData()
            => ThrowIfNotInitialized().flatData!;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected override void NoLock_Free()
        {
            base.NoLock_Free();
            flatData = null;
        }

        internal T[] GetOrInitializeData()
        {
            lock (this)
            {
                NoLock_Initialize();
                return flatData!;
            }
        }

        #region Casting Operators
        /// <summary>
        /// Explicitly converts the shaped array to a one-dimensional array.
        /// </summary>
        /// <param name="shapedArray">The shaped array to convert.</param>
        /// <returns>The one-dimensional array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator T[](DataBuffer<T> shapedArray) => (T[])shapedArray.internalData!;

        /// <summary>
        /// Explicitly converts the shaped array to a two-dimensional array.
        /// </summary>
        /// <param name="shapedArray">The shaped array to convert.</param>
        /// <returns>The two-dimensional array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator T[,](DataBuffer<T> shapedArray) => (T[,])shapedArray.internalData!;

        /// <summary>
        /// Explicitly converts the shaped array to a three-dimensional array.
        /// </summary>
        /// <param name="shapedArray">The shaped array to convert.</param>
        /// <returns>The three-dimensional array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator T[,,](DataBuffer<T> shapedArray) => (T[,,])shapedArray.internalData!;

        /// <summary>
        /// Explicitly converts the shaped array to a four-dimensional array.
        /// </summary>
        /// <param name="shapedArray">The shaped array to convert.</param>
        /// <returns>The four-dimensional array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator T[,,,](DataBuffer<T> shapedArray) => (T[,,,])shapedArray.internalData!;

        /// <summary>
        /// Explicitly converts the shaped array to a five-dimensional array.
        /// </summary>
        /// <param name="shapedArray">The shaped array to convert.</param>
        /// <returns>The five-dimensional array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator T[,,,,](DataBuffer<T> shapedArray) => (T[,,,,])shapedArray.internalData!;

        /// <summary>
        /// Explicitly converts the shaped array to a six-dimensional array.
        /// </summary>
        /// <param name="shapedArray">The shaped array to convert.</param>
        /// <returns>The six-dimensional array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator T[,,,,,](DataBuffer<T> shapedArray) => (T[,,,,,])shapedArray.internalData!;

        /// <summary>
        /// Explicitly converts the shaped array to a seven-dimensional array.
        /// </summary>
        /// <param name="shapedArray">The shaped array to convert.</param>
        /// <returns>The seven-dimensional array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator T[,,,,,,](DataBuffer<T> shapedArray) => (T[,,,,,,])shapedArray.internalData!;
        #endregion
    }
}
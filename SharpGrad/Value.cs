using Microsoft.CodeAnalysis.CSharp.Syntax;
using SharpGrad.DifEngine.SyntaxBuilder;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SharpGrad
{
    /// <summary>
    /// Base class for all values in the computation graph.
    /// </summary>
    public abstract class Value(string name, DataBuffer data, Shape shape, KindGraphNode kind) : IValue
    {
        public readonly string Name = name;

        /// <summary>
        /// The shape of the value.
        /// </summary>
        public Shape Shape { get; protected set; } = shape;

        /// <summary>
        /// If true, this value is an output of the computation graph and should be saved back to its data field.
        /// </summary>
        public bool IsOutput { get; set; } = false;

        protected readonly DataBuffer untypedData = data;
        public IReadOnlyDataBuffer Data
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => untypedData;
        }

        /// <summary>
        /// Initializes the data buffer with the specified type.
        /// </summary>
        /// <typeparam name="TType">The type to initialize the data buffer with.</typeparam>
        /// <returns>True if the data buffer was successfully initialized, false if it was already initialized.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public abstract bool InitializeData();


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal Span<TType> GetInitializedData<TType>() where TType : struct, INumber<TType>
        {
            if (untypedData is DataBuffer<TType> dataBuffer)
            {
                return dataBuffer.GetInitializedData();
            }
            throw new InvalidOperationException($"Trying to get data type {typeof(TType)}, but it is set to {untypedData.ElementType}.");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal Span<TType> GetOrInitializeData<TType>() where TType : struct, INumber<TType>
        {
            if (untypedData is DataBuffer<TType> dataBuffer)
            {
                return dataBuffer.GetOrInitializeData();
            }
            throw new InvalidOperationException($"Trying to get data type {typeof(TType)}, but it is set to {untypedData.ElementType}.");
        }

        internal DataBuffer? untypedGrad = null;
        public ReadOnlyDataBuffer Grad
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => untypedGrad ?? throw new InvalidOperationException("Gradient is not available.");
        }
        IReadOnlyDataBuffer IValue.Grad
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Grad;
        }
        public bool IsGradTypeSet
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => untypedGrad is not null;
        }

        /// <summary>
        /// Initializes the gradient buffer with the specified type if it is not already initialized. If it is already initialized with the same type, does nothing.
        /// If it is already initialized with a different type, throws an exception.
        /// </summary>
        /// <typeparam name="TGrad">The type to initialize the gradient buffer with.</typeparam>
        /// <returns>True if the gradient buffer was successfully initialized, false if it was already initialized with the same type.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the gradient buffer is already initialized with a different type.</exception>
        internal bool InitializeGrad<TGrad>() where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            if (untypedGrad is null)
            {
                DataBuffer<TGrad> newGrad = DataBuffer.Create<TGrad>(Shape);
                newGrad.Initialize();
                untypedGrad = newGrad;
                return true;
            }
            else
            {
                if (untypedGrad is DataBuffer<TGrad> dataBuffer)
                {
                    return dataBuffer.Initialize();
                }
                throw new InvalidOperationException($"Trying to set gradient type to {typeof(TGrad)}, but it is already set to {untypedGrad.ElementType}.");
            }
        }

        /// <summary>
        /// Gets the initialized gradient buffer with the specified type.
        /// </summary>
        /// <typeparam name="TGrad">The type of the gradient buffer.</typeparam>
        /// <returns>The gradient buffer.</returns>
        /// <remarks>
        /// If the gradient buffer is not initialized or is initialized with a different type, an exception is thrown.
        /// </remarks>
        public DataBuffer<TGrad> GetInitializedGradBuffer<TGrad>()
        {
            if (untypedGrad is null)
            {
                throw new InvalidOperationException("Gradient is not initialized.");
            }
            if (untypedGrad is DataBuffer<TGrad> dataBuffer)
            {
                return dataBuffer;
            }
            throw new InvalidOperationException($"Trying to get gradient type {typeof(TGrad)}, but it is set to {untypedGrad.ElementType}.");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal Span<TGrad> GetInitializedGrad<TGrad>()
        {
            if (untypedGrad is null)
            {
                throw new InvalidOperationException("Gradient is not initialized.");
            }
            if (untypedGrad is DataBuffer<TGrad> dataBuffer)
            {
                return dataBuffer.GetInitializedData();
            }
            throw new InvalidOperationException($"Trying to get gradient type {typeof(TGrad)}, but it is set to {untypedGrad.ElementType}.");
        }

        /// <summary>
        /// Gets the initialized flat data array of type T from a untyped DataBuffer.
        /// </summary>
        /// <param name="buffer">The untyped DataBuffer.</param>
        /// <returns>The already initialized flat data array of type T.</returns>
        /// <remarks>
        /// Throws an InvalidOperationException if the buffer is not of the expected type or is not initialized.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal Span<TGrad> GetOrInitializeGrad<TGrad>()
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            DataBuffer<TGrad> buffer = GetOrInitializeGradBuffer<TGrad>();
            if (!buffer.IsInitialized)
            {
                buffer.Initialize();
            }
            return buffer.GetOrInitializeData();
        }


        /// <summary>
        /// Gets or initializes the gradient buffer with the specified type.
        /// </summary>
        /// <typeparam name="TGrad">The type of the gradient buffer.</typeparam>
        /// <returns>The gradient buffer.</returns>
        /// <remarks>
        /// If the gradient buffer is already initialized with a different type, an exception is thrown.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the gradient buffer is already initialized with a different type.</exception>
        public DataBuffer<TGrad> GetOrInitializeGradBuffer<TGrad>()
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            InitializeGrad<TGrad>();
            return GetInitializedGradBuffer<TGrad>();
        }

        public bool IsGradiable { get; set; } = !kind.IsValue();

        public KindGraphNode Kind
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get;
        } = kind;

        public Type ElementType
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Data.ElementType;
        }

        protected Value(string name, Type type, Shape shape, KindGraphNode kind)
            : this(
                  name,
                  DataBuffer.Create(type, shape),
                  shape,
                  kind
            )
        { }

        protected Value(string name, Array data, Shape shape, KindGraphNode kind)
            : this(
                  name,
                  DataBuffer.Create(data, shape),
                  shape,
                  kind
            )
        { }

        public override string ToString() => Name;

        public Value<T> As<T>()
            where T : struct, INumber<T>
            => (Value<T>)this;
    }
}
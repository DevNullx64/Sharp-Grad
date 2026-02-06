using Microsoft.CodeAnalysis.CSharp.Syntax;
using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SharpGrad
{
    /// <summary>
    /// Base class for all values in the computation graph.
    /// </summary>
    public abstract class Value(string name, DataBuffer data, DataBuffer grad, Shape shape, KindGraphNode kind) : IValue
    {
        public readonly string Name = name;

        /// <summary>
        /// The shape of the value.
        /// </summary>
        public Shape Shape { get; } = shape;

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TType[] GetInitializedData<TType>() where TType : struct, INumber<TType>
        {
            if (untypedData is DataBuffer<TType> dataBuffer)
            {
                return dataBuffer.GetInitializedData();
            }
            throw new InvalidOperationException($"Trying to get data type {typeof(TType)}, but it is set to {untypedData.ElementType}.");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TType[] GetOrInitializeData<TType>() where TType : struct, INumber<TType>
        {
            if (untypedData is DataBuffer<TType> dataBuffer)
            {
                return dataBuffer.GetOrInitializeData();
            }
            throw new InvalidOperationException($"Trying to get data type {typeof(TType)}, but it is set to {untypedData.ElementType}.");
        }

        internal DataBuffer? untypedGrad = grad;
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
        internal TGrad[] GetInitializedGrad<TGrad>()
        {
            if(untypedGrad is null)
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
        internal TGrad[] GetOrInitializeGrad<TGrad>()
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => GetOrInitializeGradBuffer<TGrad>().GetInitializedData();


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
            if (untypedGrad is null)
            {
                DataBuffer<TGrad> newGrad = DataBuffer.Create<TGrad>(Shape);
                newGrad.Initialize();
                untypedGrad = newGrad;
                return newGrad;
            }
            else
            {
                if (untypedGrad is DataBuffer<TGrad> dataBuffer)
                {
                    if (!dataBuffer.IsInitialized)
                    {
                        dataBuffer.Initialize();
                    }
                    return dataBuffer;
                }
                throw new InvalidOperationException($"Trying to set gradient type to {typeof(TGrad)}, but it is already set to {untypedGrad.ElementType}.");
            }
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
                  DataBuffer.Create(type, shape),
                  shape,
                  kind
            )
        { }

        protected Value(string name, Array data, Shape shape, KindGraphNode kind)
            : this(
                  name,
                  DataBuffer.Create(data, shape),
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
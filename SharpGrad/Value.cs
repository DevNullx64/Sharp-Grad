using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;

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
        /// Gets or initializes the gradient buffer with the specified type.
        /// </summary>
        /// <typeparam name="GradType">The type of the gradient buffer.</typeparam>
        /// <returns>The gradient buffer.</returns>
        /// <remarks>
        /// If the gradient buffer is already initialized with a different type, an exception is thrown.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the gradient buffer is already initialized with a different type.</exception>
        public DataBuffer<GradType> InitializeGrad<GradType>()
            where GradType : struct, IFloatingPointIeee754<GradType>
        {
            if (untypedGrad is not null)
            {
                if (untypedGrad is DataBuffer<GradType> dataBuffer)
                {
                    return dataBuffer;
                }
                throw new InvalidOperationException($"Trying to set gradient type to {typeof(GradType)}, but it is already set to {untypedGrad.ElementType}.");
            }
            DataBuffer<GradType> newGrad = DataBuffer.Create<GradType>(Shape);
            newGrad.Initialize();
            untypedGrad = newGrad;
            return newGrad;
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
        {
            if (Data.ElementType != typeof(T))
            {
                throw new InvalidCastException($"Cannot cast Value of type {Data.ElementType} to Value<{typeof(T)}>.");
            }
            return (Value<T>)this;
        }
    }
}
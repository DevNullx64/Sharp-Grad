using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad
{
    /// <summary>
    /// Base class for all typed values in the computation graph.
    /// </summary>
    /// <typeparam name="TType">The numeric type of the value.</typeparam>
    public abstract class Value<TType> : Value, IValue<TType>
        where TType : struct, INumber<TType>
    {
        private static int InstanceCount = 0;
        public static readonly Constant<TType> E = new(TType.CreateSaturating(Math.E), "e");
        public static readonly Constant<TType> Zero = new(TType.Zero, "0");

        public Value(IReadOnlyList<Dimension> shape, string name, KindGraphNode kind)
            : base(name, typeof(TType), new Shape([.. shape.Where(e => e.Size > 1).Distinct()]), kind)
        {
            data = (DataBuffer<TType>)untypedData;
        }

        public TType this[params int[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Data[indices];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => data[indices] = value;
        }

        public TType this[params Index[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Data[indices];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => data[indices] = value;
        }

        public TType this[Dimdices indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Data[indices];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal set => data[indices] = value;
        }


        internal readonly DataBuffer<TType> data;

        public new IReadOnlyDataBuffer<TType> Data
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => data;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal DataBuffer<TType> GetInitializedDataBuffer()
        {
            if (!data.IsInitialized)
            {
                throw new InvalidOperationException($"Data buffer for value '{Name}' is not initialized.");
            }
            return data;
        }

        /// <summary>
        /// Gets the data buffer, initializing it if it hasn't been initialized yet.
        /// </summary>
        /// <returns>The initialized data buffer.</returns>
        /// <remarks>
        /// This method is used internally to ensure that the data buffer is initialized before use.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal DataBuffer<TType> GetOrInitializeDataBuffer()
        {
            data.Initialize();
            return data;
        }

        /// <summary>
        /// Gets the underlying array of the data buffer, initializing it if it hasn't been initialized yet.
        /// </summary>
        /// <returns>The initialized data array.</returns>
        /// <remarks>
        /// This method is used internally to ensure that the data buffer is initialized before accessing the underlying array.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TType[] GetInitializedDataArray()
            {
            return data.GetInitializedDataArray();
        }


        #region BASIC ARITHMETIC OPERATIONS
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> Add(Value<TType> left, Value<TType> right)
            => new(KindBinary.Add, left, right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> operator +(Value<TType> left, Value<TType> right)
            => Add(left, right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UnaryComputedValue<TType> Neg(Value<TType> operand)
            => new(KindUnary.Negate, operand);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UnaryComputedValue<TType> operator -(Value<TType> operand)
            => Neg(operand);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> Sub(Value<TType> left, Value<TType> right)
            => new(KindBinary.Subtract, left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> operator -(Value<TType> left, Value<TType> right)
            => Sub(left, right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> Mul(Value<TType> left, Value<TType> right)
            => new(KindBinary.Multiply, left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> operator *(Value<TType> left, Value<TType> right)
            => Mul(left, right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> Div(Value<TType> left, Value<TType> right)
            => new(KindBinary.Divide, left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static BinaryComputedValue<TType> operator /(Value<TType> left, Value<TType> right)
            => Div(left, right);
        #endregion

        public static implicit operator Value<TType>(TType d)
            => new Constant<TType>(d, $"v{InstanceCount++}");
    }
}
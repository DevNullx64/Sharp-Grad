using ILGPU;
using System;
using System.Numerics;

namespace SharpGrad.Formula.Internal
{
    /// <summary>
    /// Represents an index that can be empty.
    /// </summary>
    /// <remarks>
    /// The index is bounded between 0 and <typeparamref name="T"/>.MaxValue - 1.
    /// </remarks>
    public readonly struct BIndex<T> : IEquatable<BIndex<T>>, IComparable<BIndex<T>>
        where T : IUnsignedNumber<T>, IComparable<T>, IEquatable<T>
    {
        /// <summary>
        /// The minimum value of the index.
        /// </summary>
        public static readonly T MinValue = T.Zero;

        /// <summary>
        /// The maximum value of the index.
        /// </summary>
        public static readonly T MaxValue = (T)typeof(T).GetField("MaxValue").GetValue(null) - T.One;

        public static readonly T EmptyValue = T.Zero;

        public static readonly BIndex<T> Empty = new();

        // The value of the index.
        private readonly T value;

        private BIndex(T value)
        {
            this.value = value;
        }

        /// <summary>
        /// Gets a value indicating whether the index is empty.
        /// </summary>
        public readonly bool IsEmpty => value.Equals(EmptyValue);

        /// <inheritdoc/>
        public readonly int CompareTo(BIndex<T> other)
            => value.CompareTo(other.value);

        /// <inheritdoc/>
        public static bool operator <(BIndex<T> left, BIndex<T> right)
            => left.CompareTo(right) < 0;
        /// <inheritdoc/>
        public static bool operator <=(BIndex<T> left, BIndex<T> right)
            => left.CompareTo(right) <= 0;

        /// <inheritdoc/>
        public static bool operator >(BIndex<T> left, BIndex<T> right)
            => left.CompareTo(right) > 0;

        /// <inheritdoc/>
        public static bool operator >=(BIndex<T> left, BIndex<T> right)
            => left.CompareTo(right) >= 0;

        /// <inheritdoc/>
        public readonly bool Equals(BIndex<T> other)
            => value == other.value;

        /// <inheritdoc/>
        public override readonly bool Equals(object? obj)
            => obj is BIndex<T> other && Equals(other);

        /// <inheritdoc/>
        public static bool operator ==(BIndex<T> left, BIndex<T> right)
            => left.Equals(right);

        /// <inheritdoc/>
        public static bool operator !=(BIndex<T> left, BIndex<T> right)
            => !left.Equals(right);

        /// <inheritdoc/>
        public override readonly int GetHashCode()
            => value.GetHashCode();

        /// <summary>
        /// Implicitly converts a bounded index to an integer.
        /// </summary>
        public static implicit operator int(BIndex<T> index)
            => int.CreateTruncating(index.value - T.One);

        /// <summary>
        /// Implicitly converts an integer to a bounded index.
        /// </summary>
        public static implicit operator BIndex<T>(byte index)
            => new(index < 0 ? EmptyValue : T.CreateChecked(index + 1));

        public static implicit operator Index1D(BIndex<T> index)
            => index;
    }


    public readonly struct OperandIndex<T>(T value)
        where T : ISignedNumber<T>, IComparable<T>, IEquatable<T>, IBitwiseOperators<T, T, T>
    {
        public static readonly T EmptyValue = (T)typeof(T).GetField("MinValue").GetValue(null);
        public static bool GetIsEmpty(T value) => value.Equals(EmptyValue);

        public readonly T Value = value;

        public readonly bool IsOperation => Value.CompareTo(EmptyValue) < 0;
        public readonly bool FromMemory => Value.CompareTo(EmptyValue) >= 0;
        public readonly bool IsEmpty => GetIsEmpty(Value);

        public T Index => FromMemory ? Value : ~Value;
    }
}
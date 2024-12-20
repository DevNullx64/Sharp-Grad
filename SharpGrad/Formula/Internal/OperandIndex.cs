using System;

namespace SharpGrad.Formula.Internal
{
    public readonly struct OperandIndex(int value, OperandIndexSource source)
    {
        public static readonly OperandIndex Empty = new(-1, OperandIndexSource.None);
        public static readonly short MinValue = 0;
        public static readonly short MaxValue = ushort.MaxValue / (short)OperandIndexSource.MaxValue;

        private readonly ushort value = source == OperandIndexSource.None
            ? ushort.MaxValue
            : value >= MaxValue ? throw new ArgumentOutOfRangeException(nameof(value), $"Result must be less than {MaxValue}.")
            : value < 0 ? throw new ArgumentOutOfRangeException(nameof(value), "Result must be greater than or equal to 0.")
            : (ushort)((short)source * MaxValue + value);

        public readonly bool IsEmpty => value == ushort.MaxValue;
        public readonly short Value => (short)(IsEmpty ? -1 : value % MaxValue);
        public readonly OperandIndexSource Source => IsEmpty
            ? OperandIndexSource.None
            : (OperandIndexSource)(value / MaxValue);

        public override string ToString()
            => $"{Enum.GetName(typeof(OperandIndexSource), Source)}[{Value}]";
    }
}
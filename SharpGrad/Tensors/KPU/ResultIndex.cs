using System;

namespace SharpGrad.Tensors.KPU
{
    public readonly struct ResultIndex(int value, ResultIndexSource source)
    {
        public static readonly short MinValue = 0;
        public static readonly short MaxValue = ushort.MaxValue / (short)OperandIndexSource.MaxValue;

        private readonly ushort value =
              (value >= MaxValue) ? throw new ArgumentOutOfRangeException(nameof(value), $"Value must be less than {MaxValue}.")
            : (value < 0) ? throw new ArgumentOutOfRangeException(nameof(value), "Value must be greater than or equal to 0.")
            : (ushort)((short)source * MaxValue + value);

        public readonly bool IsEmpty => value == ushort.MaxValue;
        public readonly short Value => (short)(IsEmpty ? -1 : value % MaxValue);
        public readonly OperandIndexSource Source => (IsEmpty)
            ? OperandIndexSource.None
            : (OperandIndexSource)(value / MaxValue);

        public override string ToString()
            => $"{Enum.GetName(typeof(OperandIndexSource), Source)}[{Value}]";
    }

}
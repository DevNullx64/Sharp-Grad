using System;

namespace SharpGrad.Tensors.KPU
{
    public readonly struct KPUIndex
    {
        public static readonly KPUIndex Empty = new(-1, KPUIndexSource.None);
        public static readonly short MinValue = 0;
        private const ushort Shift = 14;
        public static readonly short MaxValue = (1 << Shift) - 1;

        private readonly ushort value;

        public readonly bool IsEmpty => (short)value == -1;
        public readonly short Value => (short)(IsEmpty ? -1 : value & MaxValue);
        public readonly KPUIndexSource Source => IsEmpty ? KPUIndexSource.None : (KPUIndexSource)(value >> Shift);

        public KPUIndex(int value, KPUIndexSource source)
        {
            if (value > MaxValue)
                throw new ArgumentOutOfRangeException(nameof(value), $"Value must be less than {MaxValue}.");
            if (source == KPUIndexSource.None)
                this.value = ushort.MaxValue;
            else
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), "Value must be greater than or equal to 0.");
                else
                    this.value = (ushort)((ushort)source << Shift | (ushort)value);
            }
        }
    }

}
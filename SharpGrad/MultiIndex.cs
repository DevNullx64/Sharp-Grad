using System;

namespace SharpGrad
{
    public readonly struct MultiIndex<TEnum> where TEnum : Enum
    {
        public static readonly ushort EmptyValue = ushort.MaxValue;
        public static readonly MultiIndex<TEnum> Empty = new(EmptyValue);

        private static readonly ushort count = ushort.CreateChecked(Enum.GetValues(typeof(TEnum)).Length);
        public static readonly ushort MinValue = 0;
        public static readonly ushort MaxValue = (ushort)(ushort.MaxValue / count);

        private readonly ushort @this;

        public readonly bool IsEmpty => @this == EmptyValue;

        private MultiIndex(ushort rawValue)
        {
            @this = rawValue;
        }

        public MultiIndex(TEnum category, int value)
        {
            if (value >= MinValue)
            {
                if (value <= MaxValue)
                {
                    @this = (ushort)(Convert.ToByte(category) * MaxValue + value);
                }
                else
                {
                    throw new ArgumentOutOfRangeException($"{nameof(value)} must be between {MinValue} and {MaxValue}. Got {value}.");
                }
            }
            else
                @this = EmptyValue;
        }

        public readonly TEnum Category => (TEnum)(object)(@this / MaxValue);
        public readonly int Index =>  (@this % MaxValue);
    }
}
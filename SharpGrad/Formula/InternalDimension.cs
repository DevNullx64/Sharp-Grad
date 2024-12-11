using SharpGrad.Operators;
using System;

namespace SharpGrad.Formula
{
    internal readonly struct InternalDimension(int size, SharedReduceCode reduce, SharedBroadcastCode broadcast)
    {
        public static InternalDimension Scalar => new(1, SharedReduceCode.Mean, SharedBroadcastCode.Repeat);
        public readonly int Size = size;
        private readonly byte Operations = (byte)reduce < 15 && (byte)broadcast < 15
            ? (byte)((byte)reduce & 0x0F | ((byte)broadcast & 0x0F) << 4)
            : throw new ArgumentException("Invalid reduce or broadcast code.");
        public readonly SharedReduceCode Reduce => (SharedReduceCode)(Operations & 0x0F);
        public readonly SharedBroadcastCode Broadcast => (SharedBroadcastCode)(Operations >> 4 & 0x0F);
    }

}
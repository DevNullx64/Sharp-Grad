using System;

namespace SharpGrad.Tensors
{


    public partial class KernelProcessUnit
    {
        public readonly struct ByteArgs : IEquatable<ByteArgs>
        {
            private readonly int @internal;
            public byte CacheSize => (byte)(@internal & 0xFF);
            public byte ShapeDims => (byte)((@internal >> 8) & 0xFF);
            public byte ApplyDim => (byte)((@internal >> 16) & 0xFF);
            public byte ReduceCount => (byte)((@internal >> 24) & 0xFF);

            public ByteArgs(byte cacheSize, byte shapeDims, byte applyDim, byte reduceCount)
            {
                @internal = cacheSize | (shapeDims << 8) | (applyDim << 16) | (reduceCount << 24);
            }

            public bool Equals(ByteArgs other) => @internal == other.@internal;

            public override bool Equals(object? obj)
                => obj is ByteArgs other && Equals(other);

            public override int GetHashCode()
                => @internal.GetHashCode();

            public static bool operator ==(ByteArgs left, ByteArgs right)
                => left.Equals(right);

            public static bool operator !=(ByteArgs left, ByteArgs right)
                => !(left == right);

        }
    }
}

using System;

namespace SharpGrad
{
    public readonly struct Dimension : IEquatable<Dimension>
    {
        private readonly byte _index;
        public static readonly Dimension Scalar = new(0);

        public bool IsScalar => _index == 0;

        private Dimension(byte index)
        {
            _index = index;
        }

        public readonly string Name => DimensionsPool.GetName(_index);
        public readonly int Size => DimensionsPool.GetSize(_index);

        public Dimension(string name)
        {
            byte index = DimensionsPool.IndexOf(name);
            if (index == DimensionsPool.NotFound)
            {
                throw new ArgumentException($"Dimension '{name}' does not exist.");
            }
            else
            {
                _index = index;
            }
        }
        public Dimension(string name, int size)
        {
            _index = DimensionsPool.GetOrAddDimension(name, size);
        }

        public static bool operator ==(Dimension left, Dimension right) => left._index == right._index;
        public static bool operator !=(Dimension left, Dimension right) => left._index != right._index;
        public override bool Equals(object? obj) => obj is Dimension dimension && Equals(dimension);

        public bool Equals(Dimension other) => _index == other._index;

        public override int GetHashCode() => typeof(Dimension).GetHashCode() ^ _index.GetHashCode();

        public static implicit operator Dimension(string name) => new(name);
        public static implicit operator Dimension((string name, int size) tuple) => new(tuple.name, tuple.size);
    }
}
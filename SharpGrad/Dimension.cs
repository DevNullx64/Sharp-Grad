using System;
using System.Collections.Generic;

namespace SharpGrad
{
    public readonly struct Dimension : IEquatable<Dimension>
    {
        private static readonly List<int> sizes = [0];
        private static readonly List<string> names = ["Scalar"];

        private readonly byte _index;
        public static readonly Dimension Scalar = new(0);
        private Dimension(byte index)
        {
            _index = index;
        }

        public readonly string Name => names[_index];
        public readonly int Size => sizes[_index];

        public Dimension(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Dimension name cannot be null or whitespace.");
            }
            int index = names.IndexOf(name);
            if (index >= 0)
            {
                _index = (byte)index;
            }
            else
            {
                throw new ArgumentException($"Dimension '{name}' does not exist.");
            }
        }
        public Dimension(string name, int size)
        {
            if (size < 2)
            {
                throw new ArgumentException($"Size must be greater than 1. Got {size}.");
            }
            int index = names.IndexOf(name);
            if (index >= 0)
            {
                if (sizes[index] != size)
                {
                    throw new ArgumentException($"Dimension '{name}' already exists with size {sizes[index]}, cannot redefine with size {size}.");
                }
                if(index >= byte.MaxValue)
                {
                    throw new ArgumentException($"Too many dimensions defined. Maximum is {byte.MaxValue}.");
                }
                _index = (byte)index;
            }
            else
            {
                _index = (byte)names.Count;
                names.Add(name);
                sizes.Add(size);
            }
        }

        public static bool operator ==(Dimension left, Dimension right) => left._index == right._index;
        public static bool operator !=(Dimension left, Dimension right) => left._index != right._index;
        public override bool Equals(object? obj) => obj is Dimension dimension && Equals(dimension);

        public bool Equals(Dimension other) => _index == other._index;

        public override int GetHashCode() => typeof(Dimension).GetHashCode() ^ _index.GetHashCode();
    }
}
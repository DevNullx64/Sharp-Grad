using System;

namespace SharpGrad
{
    public readonly struct Dim : IEquatable<Dim>, IComparable<Dim>
    {
        public readonly int Size;

        private Dim(int size) => Size = size;

        public int CompareTo(Dim other) => Size.CompareTo(other.Size);
        public static bool operator <(Dim left, Dim right) => left.CompareTo(right) < 0;
        public static bool operator <=(Dim left, Dim right) => left.CompareTo(right) <= 0;
        public static bool operator >(Dim left, Dim right) => left.CompareTo(right) > 0;
        public static bool operator >=(Dim left, Dim right) => left.CompareTo(right) >= 0;


        public bool Equals(Dim other) => Size == other.Size;
        public override bool Equals(object obj) => obj is Dim dim && Equals(dim);
        public static bool operator ==(Dim left, Dim right) => left.Equals(right);
        public static bool operator !=(Dim left, Dim right) => !left.Equals(right);


        public static implicit operator Dim(int size) => new(size);
        public static implicit operator int(Dim dim) => dim.Size;

        public override int GetHashCode() => HashCode.Combine(Size);

        public override string ToString() => Size.ToString();
    }
}

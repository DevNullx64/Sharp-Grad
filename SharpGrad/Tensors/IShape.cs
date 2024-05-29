using System;
using System.Collections.Generic;

namespace SharpGrad
{
    public interface IShape : IReadOnlyList<Dim>, IEquatable<Shape>
    {
        bool IsScalar { get; }
    }
}

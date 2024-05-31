using System;
using System.Collections.Generic;

namespace SharpGrad
{
    public interface IShape : IEnumerable<Dim>
    {
        Dim this[Index index] { get; }
        Dim[] this[Range range] { get; }
        int Count { get; }
        long Size { get; }
        bool IsScalar { get; }
        int GetFlattenIndex(params Index[] indices);
        Index[] GetIndices(int flattenedIndex);
    }
}

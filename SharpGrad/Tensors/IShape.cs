using System;
using System.Collections.Generic;

namespace SharpGrad
{
    public interface IShape : IEnumerable<Dim>
    {
        Dim this[int index] { get; }
        int Count { get; }
        long Size { get; }
        bool IsScalar { get; }
        int GetFlattenedIndex(params Index[] indices);
        Index[] GetIndices(int flattenedIndex);
    }
}

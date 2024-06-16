using System.Numerics;
using System;

namespace SharpGrad.Tensors
{
    public interface ITensor : IEquatable<ITensor>
    {
        Shape Shape { get; set; }
        long Length { get; }
    }

    public interface ITensor<T> : ITensor
    where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        T this[params Index[] indices] { get; }
    }
}

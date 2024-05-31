using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensorBase<T> : ICloneable
        where T : unmanaged, INumber<T>
    {
        Shape Shape { get; }
        T this[params Index[] indices] { get; set; }
        T this[Range indices] { get; set; }
    }
}

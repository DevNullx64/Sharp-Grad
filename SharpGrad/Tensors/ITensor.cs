using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public interface ITensor<T>
        where T : unmanaged, INumber<T>
        
    {
        public Shape Shape { get; }
        public long Length { get; }

        public T this[params Index[] indices] { get; set; }
        public T[,] this[params Range[] ranges] { get; set; }
    }
}

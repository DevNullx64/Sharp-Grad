using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class OpBase1<T> where T : unmanaged, INumber<T>
    {
        public static Shape ResultingShape(Shape right) => right;
    }

}
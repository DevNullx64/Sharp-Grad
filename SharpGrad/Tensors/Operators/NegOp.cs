using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class NegOp<T> : OpBase1<T>, IExecutor1<T, T> where T : unmanaged, INumber<T>
    {
        public static T Backward(T operand1, T grad) => -grad;
        public static T Exec(T operand1) => -operand1;
    }

}
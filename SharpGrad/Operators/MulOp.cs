using System;
using System.Numerics;

namespace SharpGrad.Operators
{
    internal class MulOp<T> : BaseOperation<T>, IExecBinary<T, T, T> where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Mul;
        public static string Symbol => "*";

        public static (T, T) Backward(T left, T right, T grad) => (right * grad, left * grad);
        public static T Invoke(T left, T right) => left * right;
    }
}
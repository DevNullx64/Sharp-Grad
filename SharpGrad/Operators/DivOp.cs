using System;
using System.Numerics;

namespace SharpGrad.Operators
{
    public class DivOp<T> : BaseOperation<T>, IExecBinary<T, T, T>
        where T : unmanaged, INumber<T>
    {
        public static Shape ResultingShape(Shape left, Shape right)
        {
            if (left.IsScalar)
                return right;
            if (right.IsScalar)
                return left;

            if (left != right)
                throw new NotSupportedException($"Multiplication of shapes {left} and {right} is not supported");

            return left;
        }
        public static Shape ResultingShape(Shape operand) => operand;
        public static OpCode OpCode => OpCode.Div;
        public static string Symbol => "/";

        public static (T, T) Backward(T left, T right, T grad) => (grad / right, -grad * left / (right * right));
        public static T Invoke(T left, T right) => left / right;

    }

}
using System.Numerics;

namespace SharpGrad.Operators
{
    internal class TanH<T> : BaseOperation<T>, IExecUnary<T, T>
        where T : unmanaged, INumber<T>, IHyperbolicFunctions<T>
    {
        public static Shape ResultingShape(Shape operand) => operand;
        public static OpCode OpCode => OpCode.Tanh;
        public static string Symbol => "tanh";

        public static T Backward(T left, T grad) => T.One - left * left * grad;
        public static T Invoke(T operand) => T.Tanh(operand);
    }

}
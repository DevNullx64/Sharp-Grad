using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class ExpOp<T> : OpBase1<T>, IExecutor1<T, T> where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        public static OpCode OpCode => OpCode.Exp;
        public static string Symbol => "exp";

        public static T Backward(T operand1, T grad) => grad * T.Exp(operand1);
        public static T Exec(T operand1) => T.Exp(operand1);
    }

}
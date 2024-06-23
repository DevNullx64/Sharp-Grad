using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class MulOp<T> : OpBase2<T>, IExecutor2<T, T, T> where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Mul;
        public static string Symbol => "Mul";

        public static (T, T) Backward(T operand1, T operand2, T grad) => (operand2 * grad, operand1 * grad);
        public static T Exec(T operand1, T operand2) => operand1 * operand2;
    }

}
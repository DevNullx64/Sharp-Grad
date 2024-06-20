using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class DivOp<T> : OpBase2<T>, IExecutor2<T, T, T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        public static OpCode OpCode => OpCode.Div;
        public static string Symbol => "/";

        public static (T, T) Backward(T operand1, T operand2, T grad) => (grad / operand2, -grad * operand1 / (operand2 * operand2));
        public static T Exec(T operand1, T operand2) => operand1 / operand2;
    }

}
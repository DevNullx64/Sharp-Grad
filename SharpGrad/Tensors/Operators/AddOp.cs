using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class AddOp<T> : OpBase2<T>, IExecutor2<T, T, T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        public static OpCode OpCode => OpCode.Add;

        public static (T, T) Backward(T operand1, T operand2, T grad) => (grad, grad);
        public static T Exec(T operand1, T operand2) => operand1 + operand2;
    }

}
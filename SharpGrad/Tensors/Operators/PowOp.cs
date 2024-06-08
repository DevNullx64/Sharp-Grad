using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class PowOp<T> : OpBase2<T>, IExecutor2<T, T, T> where T : unmanaged, INumber<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        public static (T, T) Backward(T operand1, T operand2, T grad) => (grad * operand2 * T.Pow(operand1, operand2 - T.One), grad * T.Log(operand1) * T.Pow(operand1, operand2));
        public static T Exec(T operand1, T operand2) => T.Pow(operand1, operand2);
    }

}
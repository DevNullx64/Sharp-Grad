using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class LogOp<T> : OpBase1<T>, IExecutor1<T, T> where T : unmanaged, INumber<T>, ILogarithmicFunctions<T>
    {
        public static T Backward(T operand1, T grad) => grad / operand1;
        public static T Exec(T operand1) => T.Log(operand1);
    }

}
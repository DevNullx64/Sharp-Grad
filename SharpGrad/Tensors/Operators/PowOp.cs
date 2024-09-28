using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class PowOp<T> : BaseOperation<T>, IExecOperation<T, T, T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        public static OpCode OpCode => OpCode.Pow;
        public static string Symbol => "^";

        public static BackwardNeedOperand BackwardOperand => BackwardNeedOperand.Both;
        public static (T, T) Backward(T left, T right, T grad) => (grad * right * T.Pow(left, right - T.One), grad * T.Log(left) * T.Pow(left, right));
        public static T Exec(T left, T right) => T.Pow(left, right);
    }

}
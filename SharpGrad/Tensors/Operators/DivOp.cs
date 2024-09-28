using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class DivOp<T> : BaseOperation<T>, IExecOperation<T, T, T>
        where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Div;
        public static string Symbol => "/";

        public static BackwardNeedOperand BackwardOperand => BackwardNeedOperand.Both;
        public static (T, T) Backward(T left, T right, T grad) => (grad / right, -grad * left / (right * right));
        public static T Exec(T left, T right) => left / right;
    }

}
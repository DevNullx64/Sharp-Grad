using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class SubOp<T> : BaseOperation<T>, IExecOperation<T, T, T>
        where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Sub;
        public static string Symbol => "-";

        public static BackwardNeedOperand BackwardOperand => BackwardNeedOperand.None;

        public static (T, T) Backward(T left, T right, T grad) => (grad, -grad);

        public static T Exec(T left, T right) => left - right;
    }

}
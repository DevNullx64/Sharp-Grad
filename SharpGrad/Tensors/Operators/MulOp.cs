using System.Numerics;
using SharpGrad.Tensors.KPU;

namespace SharpGrad.Tensors.Operators
{
    internal class MulOp<T> : BaseOperation<T>, IExecOperation<T, T, T> where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Mul;
        public static string Symbol => "Mul";

        public static BackwardNeedOperand BackwardOperand => BackwardNeedOperand.Right;
        public static (T, T) Backward(T left, T right, T grad) => (right * grad, left * grad);
        public static T Exec(T left, T right) => left * right;
    }

}
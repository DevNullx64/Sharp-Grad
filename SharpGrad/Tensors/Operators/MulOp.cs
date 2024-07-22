using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class MulOp<T> : OpBase2<T>, IExecutor2<T, T, T> where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Mul;
        public static string Symbol => "Mul";

        public static BackwardNeedOperand BackwardLeftOperand => BackwardNeedOperand.Right;
        public static T BackwardLeft(T? left, T? right, T grad) => right.Value * grad;

        public static BackwardNeedOperand BackwardRightOperand => BackwardNeedOperand.Left;
        public static T BackwardRight(T? left, T? right, T grad) => left.Value * grad;

        public static T Exec(T left, T right) => left * right;
    }

}
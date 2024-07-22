using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class DivOp<T> : OpBase2<T>, IExecutor2<T, T, T>
        where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Div;
        public static string Symbol => "/";

        public static BackwardNeedOperand BackwardLeftOperand => BackwardNeedOperand.Right;
        public static T BackwardLeft(T? left, T? right, T grad) => grad / right.Value;

        public static BackwardNeedOperand BackwardRightOperand => BackwardNeedOperand.Both;
        public static T BackwardRight(T? left, T? right, T grad) => -grad * left.Value / (right.Value * right.Value);

        public static T Exec(T left, T right) => left / right;
    }

}
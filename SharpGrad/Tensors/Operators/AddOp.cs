using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    public class AddOp<T> : OpBase2<T>, IExecutor2<T, T, T>
        where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Add;
        public static string Symbol => "+";

        public static BackwardNeedOperand BackwardLeftOperand => BackwardNeedOperand.None;
        public static T BackwardLeft(T? left, T? right, T grad) => grad;

        public static BackwardNeedOperand BackwardRightOperand => BackwardNeedOperand.None;
        public static T BackwardRight(T? left, T? right, T grad) => grad;

        public static T Exec(T left, T right) => left + right;
    }

}
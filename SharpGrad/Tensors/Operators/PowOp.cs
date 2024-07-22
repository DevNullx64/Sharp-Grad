using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class PowOp<T> : OpBase2<T>, IExecutor2<T, T, T> where T : unmanaged, INumber<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        public static OpCode OpCode => OpCode.Pow;
        public static string Symbol => "^";

        public static BackwardNeedOperand BackwardLeftOperand => BackwardNeedOperand.Both;
        public static T BackwardLeft(T? left, T? right, T grad) => grad * right.Value * T.Pow(left.Value, right.Value - T.One);

        public static BackwardNeedOperand BackwardRightOperand => BackwardNeedOperand.Both;
        public static T BackwardRight(T? left, T? right, T grad) => grad * T.Log(left.Value) * T.Pow(left.Value, right.Value);

        public static T Exec(T left, T right) => T.Pow(left, right);
    }

}
using SharpGrad.Tensors.KPU;
using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class ExpOp<T> : BaseFunction<T>, IExecUnary<T, T>
        where T : unmanaged, INumber<T>, IExponentialFunctions<T>
    {
        public static OpCode OpCode => OpCode.Exp;
        public static string Symbol => "exp";

        public static T Backward(T right, T grad) => grad * T.Exp(right);
        public static T Invoke(T right) => T.Exp(right);
    }

}
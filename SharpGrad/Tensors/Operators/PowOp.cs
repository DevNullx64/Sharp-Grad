using SharpGrad.Tensors.KPU;
using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class PowOp<T> : BaseOperation<T>, IExecBinary<T, T, T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        public static OpCode OpCode => OpCode.Pow;
        public static string Symbol => "^";

        public static (T, T) Backward(T left, T right, T grad) => (grad * right * T.Pow(left, right - T.One), grad * T.Log(left) * T.Pow(left, right));
        public static T Invoke(T left, T right) => T.Pow(left, right);
    }

}
using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class LogOp<T> : OpBase1<T>, IExecutor1<T, T> where T : unmanaged, INumber<T>, ILogarithmicFunctions<T>
    {
        public static OpCode OpCode => OpCode.Log;
        public static string Symbol => "log";

        public static T Backward(T right, T grad) => grad / right;
        public static T Exec(T right) => T.Log(right);
    }

}
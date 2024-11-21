using SharpGrad.Tensors.KPU;
using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class LogOp<T> : BaseFunction<T>, IExecUnary<T, T> where T : unmanaged, INumber<T>, ILogarithmicFunctions<T>
    {
        public static OpCode OpCode => OpCode.Log;
        public static string Symbol => "log";

        public static T Backward(T right, T grad) => grad / right;
        public static T Invoke(T right) => T.Log(right);
    }

}
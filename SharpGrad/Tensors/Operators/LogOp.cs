using System.Numerics;
using SharpGrad.Tensors.KPU;

namespace SharpGrad.Tensors.Operators
{
    internal class LogOp<T> : BaseFunction<T>, IExecUnary<T, T> where T : unmanaged, INumber<T>, ILogarithmicFunctions<T>
    {
        public static OpCode OpCode => OpCode.Log;
        public static string Symbol => "log";

        public static T Backward(T right, T grad) => grad / right;
        public static T Exec(T right) => T.Log(right);
    }

}
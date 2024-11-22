using System.Numerics;

namespace SharpGrad.Operators
{
    internal class NoneOp<T> : BaseFunction<T>, IExecUnary<T, T> where T : unmanaged, INumber<T>, ILogarithmicFunctions<T>
    {
        public static OpCode OpCode => OpCode.None;
        public static string Symbol => string.Empty;

        public static T Backward(T right, T grad) => grad;
        public static T Invoke(T right) => right;
    }

}
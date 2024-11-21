using SharpGrad.Tensors.KPU;
using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class NegOp<T> : BaseFunction<T>, IExecUnary<T, T>
        where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Neg;
        public static string Symbol => "-";

        public static T Backward(T right, T grad) => -grad;
        public static T Invoke(T right) => -right;
    }
}
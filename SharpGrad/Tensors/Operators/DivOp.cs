using SharpGrad.Tensors.KPU;
using System.Numerics;

namespace SharpGrad.Tensors.Operators
{
    internal class DivOp<T> : BaseOperation<T>, IExecBinary<T, T, T>
        where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Div;
        public static string Symbol => "/";

        public static (T, T) Backward(T left, T right, T grad) => (grad / right, -grad * left / (right * right));
        public static T Invoke(T left, T right) => left / right;
    }

}
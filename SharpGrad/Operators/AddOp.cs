using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Operators
{
    public class AddOp<T> : BaseOperation<T>, IExecBinary<T, T, T>
        where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Add;
        public static string Symbol => "+";


        public static (T, T) Backward(T left, T right, T grad) => (grad, grad);
        public static T Invoke(T left, T right) => left + right;

    }

}
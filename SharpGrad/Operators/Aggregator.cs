using System;
using System.Numerics;

namespace SharpGrad.Operators
{
    [Obsolete("Use TensorReduce instead")]
    internal class Aggregator<T> where T : unmanaged, INumber<T>
    {
        public static Shape ResultingShape(Shape operand1) => operand1[..^1];
    }

}
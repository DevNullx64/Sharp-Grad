using System.Numerics;

namespace SharpGrad.Operators
{
    internal class SumOp<T> : Aggregator<T>, IAggregator<T, T> where T : unmanaged, INumber<T>
    {
        public static OpCode OpCode => OpCode.Sum;
        public static string Symbol => "Σ";

        public static T Exec(T[] rights)
        {
            T sum = rights[0];
            for (int i = 1; i < rights.Length; i++)
                sum += rights[i];
            return sum;
        }

        public static T[] Backward(T[] rights, T grad)
        {
            T[] grads = new T[rights.Length];
            for (int i = 0; i < rights.Length; i++)
                grads[i] = grad;
            return grads;
        }
    }
}
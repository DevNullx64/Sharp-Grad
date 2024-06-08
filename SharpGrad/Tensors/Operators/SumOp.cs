using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors.Operators
{
    internal class SumOp<T> : Aggregator<T>, IAggregator<T, T> where T : unmanaged, INumber<T>
    {
        public static T Exec(T[] operand1)
        {
            T sum = operand1[0];
            for (int i = 1; i < operand1.Length; i++)
                sum += operand1[i];
            return sum;
        }

        public static T[] Backward(T[] operand1, T grad)
        {
            T[] grads = new T[operand1.Length];
            for (int i = 0; i < operand1.Length; i++)
                grads[i] = grad;
            return grads;
        }
    }
}
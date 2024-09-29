using System.Collections;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors.KPU
{
    public abstract class KpuScrip<T> : IReadOnlyList<OperationKPU>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        /// <summary>
        /// Find the next operation that use of a tensor in a list of tensors.
        /// </summary>
        /// <param name="tensor">The tensor to find the next use of.</param>
        /// <param name="tensors">The list of tensors.</param>
        /// <param name="starting">The index to start searching from.</param>
        /// <returns>The index of the next operation that use of the tensor after the starting index.</returns>
        protected static int NextUse(Tensor<T> tensor, List<Tensor<T>> tensors, int starting)
        {
            for (int j = starting + 1; j < tensors.Count; j++)
            {
                var t = tensors[j];
                if (t.OperandCount == 1)
                {
                    ITensorOperation1<T> operation1 = (ITensorOperation1<T>)t;
                    if (operation1.Operand.Equals(tensor))
                        return j;
                }
                else if (t.OperandCount == 2)
                {
                    ITensorOperation2<T> operation2 = (ITensorOperation2<T>)t;
                    if (operation2.Operand1.Equals(tensor))
                        return j;
                    if (operation2.Operand2.Equals(tensor))
                        return j;
                }
            }
            return -1;
        }

        /// <summary>
        /// Count the number of times a tensor is used after a given index.
        /// </summary>
        /// <typeparam name="T">The type of the tensor.</typeparam>
        /// <param name="tensor">The tensor to count the usage of.</param>
        /// <param name="tensors">The list of tensors.</param>
        /// <param name="starting">The index to start counting from.</param>
        /// <returns>Operation that uses tensort twice is counted once.</returns>
        protected int UsageCount(Tensor<T> tensor, List<Tensor<T>> tensors, int starting)
        {
            int count = 0;
            int next;
            while ((next = NextUse(tensor, tensors, starting)) != -1)
            {
                count++;
                starting = next;
            }
            return count;
        }


        protected readonly List<OperationKPU> operations = [];
        public OperationKPU this[int index] => operations[index];

        protected readonly List<Tensor<T>> operands = [];
        public IReadOnlyList<Tensor<T>> Datas => operands;
        public byte CacheSize { get; protected set; }

        public int Count => operations.Count;

        public IEnumerator<OperationKPU> GetEnumerator() => ((IEnumerable<OperationKPU>)operations).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => operations.GetEnumerator();
    }
}
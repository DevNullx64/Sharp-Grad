using SharpGrad.Tensors.KPU;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public abstract class KpuScrip<T> : IReadOnlyList<OperationKPU>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
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
            for (int j = starting + 1; j < tensors.Count; j++)
            {
                var t = tensors[j];
                if (t.OperandCount == 1 && t is ITensorOperation1<T> operation1)
                {
                    if (operation1.Operand.Equals(tensor))
                        count++;
                }
                else if (t.OperandCount == 2 && t is ITensorOperation2<T> operation2)
                {
                    if (operation2.Operand1.Equals(tensor))
                        count++;
                    else if (operation2.Operand2.Equals(tensor))
                        count++;
                }
            }
            return count;
        }

        protected static bool WillBeUsed(Tensor<T> tensor, List<Tensor<T>> tensors, int starting)
        {
            for (int j = starting + 1; j < tensors.Count; j++)
            {
                var t = tensors[j];
                if (t.OperandCount == 1 && t is ITensorOperation1<T> operation1)
                {
                    if (operation1.Operand.Equals(tensor))
                        return true;
                }
                else if (t.OperandCount == 2 && t is ITensorOperation2<T> operation2)
                {
                    if (operation2.Operand1.Equals(tensor))
                        return true;
                    if (operation2.Operand2.Equals(tensor))
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Store a tensor in the first available register. Otherwise, add it to the list of registers.
        /// </summary>
        /// <typeparam name="T">The type of the tensor.</typeparam>
        /// <param name="registers">The list of registers.</param>
        /// <param name="tensor">The tensor to store.</param>
        /// <returns>The index of the register where the tensor is stored.</returns>
        /// <remarks>This is the index in regiters, not the KPU index.</remarks>
        protected short Store(List<Tensor<T>?> registers, Tensor<T> tensor)
        {
            return (short)registers.Insert(tensor);
        }


        protected readonly List<OperationKPU> operations = [];
        public OperationKPU this[int index] => operations[index];

        protected readonly List<Tensor<T>> datas = [];
        public IReadOnlyList<Tensor<T>> Datas => datas;
        public byte CacheSize { get; protected set; }

        public int Count => operations.Count;

        public IEnumerator<OperationKPU> GetEnumerator() => ((IEnumerable<OperationKPU>)operations).GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => operations.GetEnumerator();
    }
}
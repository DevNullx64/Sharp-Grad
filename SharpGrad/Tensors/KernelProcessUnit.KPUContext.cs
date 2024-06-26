using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{


    public partial class KernelProcessUnit
    {
        public class KPUContext<T>
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            // contains all input tensors
            private readonly List<ITensor<T>> datas;
            public IReadOnlyList<ITensor<T>> Datas => datas;

            // contains all operations
            private readonly List<ITensor<T>> operations;
            public IReadOnlyList<ITensor<T>> Operations => operations;

            // contains the registers
            private readonly List<ITensor<T>?> registers = [];
            public IReadOnlyList<ITensor<T>?> Registers => registers;

            // contains the script
            private readonly List<OperationKPU> script = [];
            public IReadOnlyList<OperationKPU> Script => script;

            private bool IsLastUse(ITensor<T> tensor)
            {
                for (int j = current + 1; j < tensors.Count; j++)
                    if (tensor.Equals(tensors[j]))
                        return false;
                return true;
            }

            private int current = -1;
            private List<ITensor<T>> tensors;

            public KPUContext(List<ITensor<T>> tensors)
            {
                datas = [];
                operations = [];
                this.tensors = tensors;

                for (current = 0; current < tensors.Count; current++)
                {
                    Add(tensors[current]);
                }
            }

            private void Add(ITensor<T> tensor)
            {
                switch (tensor)
                {
                    case TensorData<T> data:
                        Add(data);
                        break;
                    case ITensorReduce<T> operationR:
                        Add(operationR);
                        break;
                    case ITensorOperation1<T> operation1:
                        Add(operation1);
                        break;
                    case ITensorOperation2<T> operation2:
                        Add(operation2);
                        break;
                    default:
                        throw new InvalidOperationException($"Unknown node type {tensor.GetType()}");
                }
            }

            private void Add(TensorData<T> data)
            {
                datas.Add(data);
            }

            private void Add(ITensorReduce<T> operationR)
            {
                if (!operations.Contains(operationR))
                {
                    operations.Add(operationR);
                    Add(operationR.Operand1);
                }
                throw new NotImplementedException();
            }
            private void Add(ITensorOperation1<T> operation1)
            {
                throw new NotImplementedException();
            }

            private short Store(ITensor<T> operand)
            {
                int iOp = registers.IndexOf(operand);
                if (iOp != -1)
                    return (short)(-iOp - 1);

                iOp = registers.IndexOf(null);
                if (iOp == -1)
                {
                    iOp = -registers.Count -1;
                    if (operand.Depth == 0)
                        script.Add(new OperationKPU(OpCode.Store, (short)iOp, (short)datas.IndexOf(operand)));
                    registers.Add(operand);
                }
                else
                {
                    iOp = -iOp - 1;
                    if (operand.Depth == 0)
                        script.Add(new OperationKPU(OpCode.Store, (short)iOp, (short)datas.IndexOf(operand)));
                    registers[iOp] = operand;
                }

                return (short)iOp;
            }
            private short GetOrStrore(ITensor<T> operand)
            {
                int iOp1 = registers.IndexOf(operand);
                if (iOp1 == -1)
                {
                    if (operand.Depth == 0 && IsLastUse(operand))
                    {
                        iOp1 = datas.IndexOf(operand);
                    }
                    else
                    {
                        // find the first empty register
                        int iOp = registers.IndexOf(null);
                        // if no empty register is found, create a new one
                        if (iOp == -1)
                            iOp = registers.Count;
                        // compute KPU index
                        iOp = -iOp - 1;
                        if(operand.Depth == 0)
                            script.Add(new OperationKPU(OpCode.Store, (short)iOp, (short)datas.IndexOf(operand)));
                        iOp1 = iOp;
                        registers.Add(operand);
                    }
                }
                else
                {
                    if (IsLastUse(operand))
                    {
                        registers[iOp1] = null;
                    }
                    iOp1 = -iOp1 - 1;
                }
                return (short)iOp1;
            }

            private void Add(ITensorOperation2<T> operation2)
            {
                var operand1 = operation2.Operand1;
                var operand2 = operation2.Operand2;
                short iOp1;
                int iOp2;

                if (operand1.Depth == 0 && operand2.Depth == 0)
                {
                    // The two operand are data tensors
                    iOp1 = GetOrStrore(operand1);
                    iOp2 = GetOrStrore(operand2);
                }
                else if (operand1.Depth == 0)
                {
                    // The first operand is a data tensor and the second operand is an operation
                }
                else if (operand2.Depth == 0)
                {
                    // The first operand is an operation and the second operand is a data tensor
                }
                else
                {
                    // The two operands are operations
                }
            }
        }

        public class KpuScript<T>: IReadOnlyList<OperationKPU>
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            private readonly OperationKPU[] operations = [];
            public OperationKPU this[int index] => operations[index];

            private readonly TensorData<T>[] datas = [];
            public IReadOnlyList<TensorData<T>> Datas => datas;
            public readonly int RegistersCount;

            public int Count => operations.Length;

            public KpuScript(IList<OperationKPU> operations, IList<ITensor<T>> datas, int registersCount)
            {
                this.operations = [.. operations];
                this.datas = datas.Cast<TensorData<T>>().ToArray();
                RegistersCount = registersCount;
            }

            public IEnumerator<OperationKPU> GetEnumerator() => ((IEnumerable<OperationKPU>)operations).GetEnumerator();
            IEnumerator IEnumerable.GetEnumerator() => operations.GetEnumerator();
        }

        /// <summary>
        /// Count the number of times a tensor is used after a given index.
        /// </summary>
        /// <typeparam name="T">The type of the tensor.</typeparam>
        /// <param name="tensor">The tensor to count the usage of.</param>
        /// <param name="tensors">The list of tensors.</param>
        /// <param name="starting">The index to start counting from.</param>
        /// <returns>The number of times the tensor is used.</returns>
        private int UsageCount<T>(ITensor<T> tensor, List<Tensor<T>> tensors, int starting)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            int count = 0;
            for (int j = starting + 1; j < tensors.Count; j++)
            {
                var t = tensors[j];
                if (t is ITensorOperation1<T> operation1)
                    if (operation1.Operand1.Equals(tensor))
                        count++;
                    else if (t is ITensorOperation2<T> operation2)
                    {
                        if (operation2.Operand1.Equals(tensor))
                            count++;
                        if (operation2.Operand2.Equals(tensor))
                            count++;
                    }
                    else if (t is ITensorReduce<T> operationR)
                        throw new NotImplementedException();
            }
            return count;
        }

        /// <summary>
        /// Store a tensor in the first available register. Otherwise, add it to the list of registers.
        /// </summary>
        /// <typeparam name="T">The type of the tensor.</typeparam>
        /// <param name="registers">The list of registers.</param>
        /// <param name="tensor">The tensor to store.</param>
        /// <returns>The index of the register where the tensor is stored.</returns>
        /// <remarks>This is the index in regiters, not the KPU index.</remarks>
        private int Store<T>(List<ITensor<T>?> registers, ITensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            for (int i = 0; i < registers.Count; i++)
            {
                if (registers[i] == null)
                {
                    registers[i] = tensor;
                    return i;
                }
            }
            registers.Add(tensor);
            return registers.Count - 1;
        }

        public KpuScript<T> GetKpuScript<T>(ITensor<T> tensors)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            List<ITensor<T>> datas = [];
            List<ITensor<T>> operations = [];
            List<ITensor<T>?> registers = [];
            List<OperationKPU> script = [];

            var topo = tensors.DepthFirstSearch()
                .OrderBy(e => e.Value.Index)
                .Select(e => e.Value.Tensor)
                .ToList();

            for (int i = 0; i < topo.Count; i++)
            {
                var tensor = topo[i];
                if (tensor.Depth == 0)
                {
                    // Add the data tensor to the list of data tensors only if it is not already present
                    if (!datas.Contains(tensor))
                        datas.Add(tensor);

                    // If the data tensor is used more than once, store it in a register
                    if (UsageCount(tensor, topo, i) > 1)
                    {
                        short iOp1 = (short)datas.IndexOf(tensor);
                        short iResult = (short)(-Store(registers, tensor) - 1);
                        script.Add(new OperationKPU(OpCode.Store, iResult, iOp1));
                        continue;
                    }
                }
                else
                {
                    if (registers.Contains(tensor))
                        continue;
                    if (tensor.OperandCound == 1 && tensor is ITensorOperation1<T> operation1)
                    {
                        // Operation result or stored data should contains the operand
                        short iOp1 = (short)registers.IndexOf(operation1.Operand1);
                        if (iOp1 == -1)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)datas.IndexOf(operation1.Operand1);
                            // If not, something is wrong !
                            Debug.Assert(iOp1 != -1, $"Index {i} ({operation1}) : Operand 1 {operation1.Operand1} not found.");
                        }
                        else
                        {
                            // Compute the KPU register index
                            iOp1 = (short)(-iOp1 - 1);

                            // If the operand is not used anymore, free the register
                            if (UsageCount(operation1.Operand1, topo, i) == 0)
                                registers[iOp1] = null;
                        }

                        short iResult = (short)(-Store(registers, operation1) - 1);
                        script.Add(new OperationKPU(operation1.OpCode, iResult, iOp1));
                    }
                    else if (tensor.OperandCound == 2 && tensor is ITensorOperation2<T> operation2)
                    {
                        // Operation result or stored data should contains the first operand
                        short iOp1 = (short)registers.IndexOf(operation2.Operand1);
                        if (iOp1 == -1)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)datas.IndexOf(operation2.Operand1);
                            // If not, something is wrong !
                            Debug.Assert(iOp1 != -1, $"Index {i} ({operation2}) : Operand 1 {operation2.Operand1} not found.");
                        }
                        else
                        {
                            // Compute the KPU register index
                            iOp1 = (short)(-iOp1 - 1);

                            // If the operand is not used anymore, free the register
                            if (UsageCount(operation2.Operand1, topo, i) == 0)
                                registers[iOp1] = null;
                        }

                        // Operation result or stored data should contains the second operand
                        short iOp2 = (short)registers.IndexOf(operation2.Operand2);
                        if (iOp2 == -1)
                        {
                            // If not, it's a data used only once
                            iOp2 = (short)datas.IndexOf(operation2.Operand2);
                            // If not, something is wrong !
                            Debug.Assert(iOp2 != -1, $"Index {i} ({operation2}) : Operand 2 {operation2.Operand2} not found.");
                        }
                        else
                        {
                            // Compute the KPU register index
                            iOp2 = (short)(-iOp2 - 1);

                            // If the operand is not used anymore, free the register
                            if (UsageCount(operation2.Operand2, topo, i) == 0)
                                registers[iOp2] = null;
                        }

                        short iResult = (short)(-Store(registers, operation2) - 1);
                        script.Add(new OperationKPU(operation2.OpCode, iResult, iOp1, iOp2));
                    }
                    else if (tensor.OperandCound == -1 && tensor is ITensorReduce<T> operationR)
                    {
                        throw new NotImplementedException();
                    }
                }
            }

            return new(script, datas, registers.Count);
        }
    }
}

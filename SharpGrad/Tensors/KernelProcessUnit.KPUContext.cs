﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{


    public partial class KernelProcessUnit
    {
        public class KpuScript<T> : IReadOnlyList<OperationKPU>
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
        /// <returns>Operation that uses tensort twice is counted once.</returns>
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
                        else if (operation2.Operand2.Equals(tensor))
                            count++;
                    }
                    else if (t is ITensorReduce<T> operationR)
                        throw new NotImplementedException();
            }
            return count;
        }

        private static bool WillBeUsed<T>(ITensor<T> tensor, List<Tensor<T>> tensors, int starting)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            for (int j = starting + 1; j < tensors.Count; j++)
            {
                var t = tensors[j];
                if (t is ITensorOperation1<T> operation1)
                    if (operation1.Operand1.Equals(tensor))
                        return true;
                    else if (t is ITensorOperation2<T> operation2)
                    {
                        if (operation2.Operand1.Equals(tensor))
                            return true;
                        if (operation2.Operand2.Equals(tensor))
                            return true;
                    }
                    else if (t is ITensorReduce<T> operationR)
                        throw new NotImplementedException();
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

        /// <summary>
        /// A collection of registers accessible by index or tensor.
        /// </summary>
        /// <typeparam name="T">The type of the tensor.</typeparam>
        /// <remarks>Registers are stored in a list and indexed by a dictionary.</remarks>
        private readonly struct Registers<T> : IReadOnlyList<ITensor<T>?>, IReadOnlyDictionary<ITensor<T>, int>
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            private readonly List<ITensor<T>?> listRegisters = [];
            private readonly Dictionary<ITensor<T>, int> indexedRegisters = new();
            public Registers()
            {  }

            /// <summary>
            /// Get the tensor stored in the register at the specified index.
            /// </summary>
            /// <param name="index">The index of the register.</param>
            /// <returns>The tensor stored in the register.</returns>
            public ITensor<T>? this[int index] => listRegisters[index];

            /// <summary>
            /// Get the tensor stored in the register at the specified index.
            /// </summary>
            /// <param name="index">The index of the register.</param>
            /// <returns>The tensor stored in the register.</returns>
            public ITensor<T>? this[Index index] => listRegisters[index];

            public int this[ITensor<T> key] => indexedRegisters[key];

            public int Count => listRegisters.Count;

            public IEnumerable<ITensor<T>> Keys => indexedRegisters.Keys;

            public IEnumerable<int> Values => indexedRegisters.Values;

            public int IndexOf(ITensor<T>? item) => listRegisters.IndexOf(item);

            public bool Store(ITensor<T> tensor, int usageCount)
            {
                if (ContainsKey(tensor))
                    return false;

                int i = IndexOf(null);
                if (i == -1)
                {
                    i = Count;
                    listRegisters.Add(tensor);
                }
                else
                {
                    listRegisters[i] = tensor;
                }

                indexedRegisters.Add(tensor, usageCount);
                return true;
            }

            public bool ContainsKey(ITensor<T> key) => indexedRegisters.ContainsKey(key);

            public bool Use(ITensor<T> key)
            {
                if (!indexedRegisters.ContainsKey(key))
                    return false;

                indexedRegisters[key] = indexedRegisters[key] - 1;
                if (indexedRegisters[key] == 0)
                {
                    listRegisters[IndexOf(key)] = null;
                    indexedRegisters.Remove(key);
                }
                return true;
            }

            public bool Use(int i)
            {
                var reg = listRegisters[i];
                if (reg == null)
                    return false;
                else
                {
                    indexedRegisters[reg] = indexedRegisters[reg] - 1;
                    if (indexedRegisters[reg] == 0)
                    {
                        listRegisters[i] = null;
                        indexedRegisters.Remove(reg);
                    }
                    return true;
                }
            }

            public IEnumerator<ITensor<T>> GetEnumerator() => listRegisters.GetEnumerator();

            public bool TryGetValue(ITensor<T> key, [MaybeNullWhen(false)] out int value) => indexedRegisters.TryGetValue(key, out value);
            IEnumerator IEnumerable.GetEnumerator() => listRegisters.GetEnumerator();

            IEnumerator<KeyValuePair<ITensor<T>, int>> IEnumerable<KeyValuePair<ITensor<T>, int>>.GetEnumerator() => indexedRegisters.GetEnumerator();

            internal IEnumerable<ITensor<T>?> Reverse() => ((IEnumerable<ITensor<T>?>)listRegisters).Reverse();
        }

        /// <summary>
        /// A source of data tensors and/or registers.
        /// </summary>
        /// <typeparam name="T">The type of the tensor.</typeparam>
        /// <remarks>Registers are stored with negative indices. Data tensors are stored with positive indices.</remarks>
        private readonly struct DataSource<T>: IReadOnlyList<ITensor<T>?>
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            // Contains the registers
            public readonly Registers<T> Registers = new();
            // Contains the data tensors
            public readonly List<ITensor<T>> Datas = [];

            public DataSource(ITensor<T> datas)
            {
                Datas.Add(datas);
            }

            /// <summary>
            /// Get the tensor at the specified index in the data part of the source.
            /// </summary>
            /// <param name="index">The index of the tensor.</param>
            /// <returns>The tensor at the specified index.</returns>
            /// <remarks>Index should be positive.</remarks>
            public ITensor<T>? GetData(int index) => Datas[index];

            /// <summary>
            /// Get the tensor at the specified index in the register part of the source.
            /// </summary>
            /// <param name="index">The index of the tensor.</param>
            /// <returns>The tensor at the specified index.</returns>
            /// <remarks>Index should be negative.</remarks>
            public ITensor<T>? GetRegister(int index) => Registers[-index - 1];

            /// <summary>
            /// Get the tensor at the specified index in the source.
            /// </summary>
            /// <param name="index">The index of the tensor.</param>
            /// <returns>The tensor at the specified index.</returns>
            /// <remarks>Index can be positive or negative.A positive index is used to access the data tensors. A negative index is used to access the registers.</remarks>
            public ITensor<T>? this[int index] => index >= 0 ? GetData(index) : GetRegister(index);

            /// <summary>
            /// Returns the upper bound of the source.
            /// </summary>
            /// <value>The upper bound of the source.</value>
            /// <remarks>The upper bound is the number of data tensors in the source.</remarks>
            public int UppderBound => Datas.Count;

            /// <summary>
            /// Returns the lower bound of the source.
            /// </summary>
            /// <value>The lower bound of the source.</value>
            /// <remarks>The lower bound is the negative number of registers in the source.</remarks>
            public int LowerBound => -Registers.Count - 1;

            /// <inheritdoc/>
            public int Count => Datas.Count + Registers.Count;

            /// <inheritdoc/>
            public IEnumerator<ITensor<T>?> GetEnumerator()
            {
                foreach (var register in Registers.Reverse())
                    yield return register;
                foreach (var data in Datas)
                    yield return data;
            }
            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
        }

        public KpuScript<T> GetKpuScript<T>(ITensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var topo = tensor.DepthFirstSearch()
                .OrderBy(e => e.Value.Index)
                .Select(e => e.Value.Tensor)
                .ToList();

            TensorData<T> result = new("Result", topo[^1].Shape);
            // Add the result tensor to the list at the beginning (index 0)
            List<ITensor<T>> datas = [result];
            List<ITensor<T>> operations = [];
            List<ITensor<T>?> registers = [];
            List<OperationKPU> script = [];

            for (int i = 0; i < topo.Count; i++)
            {
                var t = topo[i];
                if (t.Depth == 0)
                {
                    // Add the data tensor to the list of data tensors only if it is not already present
                    if (!datas.Contains(t))
                        datas.Add(t);

                    // If the data tensor is used more than once, store it in a register
                    if (UsageCount(t, topo, i) > 1)
                    {
                        short iOp1 = (short)datas.IndexOf(t);
                        short iResult = (short)(-Store(registers, t) - 1);
                        script.Add(new OperationKPU(OpCode.Store, iResult, iOp1));
                        continue;
                    }
                }
                else
                {
                    if (registers.Contains(t))
                        continue;

                    OpCode opCode;
                    short iOp1;
                    short iOp2;
                    short iResult;
                    if (t.OperandCound == 1 && t is ITensorOperation1<T> operation1)
                    {
                        opCode = operation1.OpCode;

                        // Operation result or stored data should contains the operand
                        iOp1 = (short)registers.IndexOf(operation1.Operand1);
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
                            if (!WillBeUsed(operation1.Operand1, topo, i))
                                registers[iOp1] = null;
                        }
                        iOp2 = OperationKPU.NoOperand;

                        iResult = (short)Store(registers, operation1);
                    }
                    else if (t.OperandCound == 2 && t is ITensorOperation2<T> operation2)
                    {
                        opCode = operation2.OpCode;

                        // Operation result or stored data should contains the first operand
                        iOp1 = (short)registers.IndexOf(operation2.Operand1);
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
                            if (!WillBeUsed(operation2.Operand1, topo, i))
                                registers[iOp1] = null;
                        }

                        // Operation result or stored data should contains the second operand
                        iOp2 = (short)registers.IndexOf(operation2.Operand2);
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
                            if (!WillBeUsed(operation2.Operand2, topo, i))
                                registers[iOp2] = null;
                        }

                        // Store result in a register if not the last operation. Otherwise, store it in datas[0] which is the result tensor
                        iResult = (short)Store(registers, operation2);
                    }
                    else //if (t.OperandCound == -1 && t is ITensorReduce<T> operationR)
                    {
                        throw new NotImplementedException();
                    }

                    iResult = (short)(i != topo.Count - 1 ? -iResult - 1 : 0);
                    script.Add(new OperationKPU(opCode, iResult, iOp1, iOp2));
                }
            }

            return new(script, datas, registers.Count);
        }
    }
}
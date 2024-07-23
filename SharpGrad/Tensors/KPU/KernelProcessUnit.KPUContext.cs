using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{


    public partial class KernelProcessUnit
    {
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

        private static bool WillBeUsed<T>(ITensor<T> tensor, List<Tensor<T>> tensors, int starting)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
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
        private short Store<T>(List<ITensor<T>?> registers, ITensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            for (short i = 0; i < registers.Count; i++)
            {
                if (registers[i] == null)
                {
                    registers[i] = tensor;
                    return i;
                }
            }
            registers.Add(tensor);
            return (short)(registers.Count - 1);
        }

        public KpuScript<T> GetKpuScript<T>(ITensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var topo = tensor.DepthFirstSearch()
                .OrderBy(e => e.Value.Index)
                .Select(e => e.Value.Tensor)
                .ToList();

            // Add the result tensor to the list at the beginning (index 0)
            List<ITensor<T>> datas = [];
            List<ITensor<T>> operations = [];
            List<ITensor<T>?> registers = [];
            List<OperationKPU> script = [];

            for (int i = 0; i < topo.Count; i++)
            {
                var t = topo[i];
                OpCode opCode;
                short iOp1;
                short iOp2;

                if (t.Depth == 0)
                {
                    // Add the data tensor to the list of data tensors only if it is not already present
                    if (!datas.Contains(t))
                        datas.Add(t);


                    int uCount = UsageCount(t, topo, i);
                    // Skip if the data tensor is used only once
                    if (uCount == 1)
                        continue;

                    opCode = OpCode.Store;
                    iOp1 = (short)datas.IndexOf(t);
                    iOp2 = OperationKPU.NoOperand;
                }
                else
                {
                    if (registers.Contains(t))
                        continue;

                    if (t.OperandCount == 1 && t is ITensorOperation1<T> operation1)
                    {
                        opCode = operation1.OpCode;

                        // Operation result or stored data should contains the operand
                        iOp1 = (short)registers.IndexOf(operation1.Operand);
                        if (iOp1 == -1)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)datas.IndexOf(operation1.Operand);
                            // If not, something is wrong !
                            if(iOp1 == -1)
                                throw new Exception($"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the register
                            if (!WillBeUsed(operation1.Operand, topo, i))
                                registers[iOp1] = null;
                            // Compute the KPU register index
                            iOp1 = (short)(-iOp1 - 1);
                        }
                        iOp2 = OperationKPU.NoOperand;
                    }
                    else if (t.OperandCount == 2 && t is ITensorOperation2<T> operation2)
                    {
                        opCode = operation2.OpCode;

                        // Operation result or stored data should contains the first operand
                        iOp1 = (short)registers.IndexOf(operation2.Operand1);
                        if (iOp1 == -1)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)datas.IndexOf(operation2.Operand1);
                            // If not, something is wrong !
                            if (iOp1 == -1)
                                throw new Exception($"Index {i} ({operation2}) : Operand 1 {operation2.Operand1} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the register
                            if (!WillBeUsed(operation2.Operand1, topo, i))
                                registers[iOp1] = null;
                            // Compute the KPU register index
                            iOp1 = (short)(-iOp1 - 1);
                        }

                        // Operation result or stored data should contains the second operand
                        iOp2 = (short)registers.IndexOf(operation2.Operand2);
                        if (iOp2 == -1)
                        {
                            // If not, it's a data used only once
                            iOp2 = (short)datas.IndexOf(operation2.Operand2);
                            // If not, something is wrong !
                            if (iOp2 == -1)
                                throw new Exception($"Index {i} ({operation2}) : Operand 2 {operation2.Operand2} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the register
                            if (!WillBeUsed(operation2.Operand2, topo, i))
                                registers[iOp2] = null;
                            // Compute the KPU register index
                            iOp2 = (short)(-iOp2 - 1);
                        }
                    }
                    else
                        throw new NotSupportedException($"Operation {t} not supported.");
                }

                // If the operation is not the last one, the result is stored in a register. Otherwise, it's the final result.
                short iResult = (short)(i != topo.Count - 1 ? -Store(registers, t) - 1 : OperationKPU.NoOperand);
                script.Add(new OperationKPU(opCode, iResult, iOp1, iOp2));
            }

            return new(script, datas, (byte)registers.Count);
        }
    }
}

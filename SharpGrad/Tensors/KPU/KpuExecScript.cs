﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class KpuExecScript<T> : KpuScrip<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        public KpuExecScript(ITensor<T> tensor)
        {
            var topo = tensor.DepthFirstSearch()
                .OrderBy(e => e.Value.Index)
                .Select(e => e.Value.Tensor)
                .ToList();

            List<ITensor<T>> dataList = [];
            List<ITensor<T>?> registerList = [];

            for (int i = 0; i < topo.Count; i++)
            {
                var t = topo[i];
                OpCode opCode;
                short iOp1;
                short iOp2;

                if (t.Depth == 0)
                {
                    // Add the data tensor to the list of data tensors only if it is not already present
                    if (!dataList.Contains(t))
                        dataList.Add(t);


                    int uCount = UsageCount(t, topo, i);
                    // Skip caching if the data tensor is used only once
                    if (uCount == 1)
                        continue;

                    opCode = OpCode.Store;
                    iOp1 = (short)dataList.IndexOf(t);
                    iOp2 = OperationKPU.NoOperand;
                }
                else
                {
                    if (registerList.Contains(t))
                        continue;

                    if (t.OperandCount == 1 && t is ITensorOperation1<T> operation1)
                    {
                        opCode = operation1.OpCode;

                        // Operation result or stored data should contains the operand
                        iOp1 = (short)registerList.IndexOf(operation1.Operand);
                        if (iOp1 == -1)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)dataList.IndexOf(operation1.Operand);
                            // If not, something is wrong !
                            if (iOp1 == -1)
                                throw new Exception($"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the register
                            if (!WillBeUsed(operation1.Operand, topo, i))
                                registerList[iOp1] = null;
                            // Compute the KPU register index
                            iOp1 = (short)(-iOp1 - 1);
                        }
                        iOp2 = OperationKPU.NoOperand;
                    }
                    else if (t.OperandCount == 2 && t is ITensorOperation2<T> operation2)
                    {
                        opCode = operation2.OpCode;

                        // Operation result or stored data should contains the first operand
                        iOp1 = (short)registerList.IndexOf(operation2.Operand1);
                        if (iOp1 == -1)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)dataList.IndexOf(operation2.Operand1);
                            // If not, something is wrong !
                            if (iOp1 == -1)
                                throw new Exception($"Index {i} ({operation2}) : Operand 1 {operation2.Operand1} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the register
                            if (!WillBeUsed(operation2.Operand1, topo, i))
                                registerList[iOp1] = null;
                            // Compute the KPU register index
                            iOp1 = (short)(-iOp1 - 1);
                        }

                        // Operation result or stored data should contains the second operand
                        iOp2 = (short)registerList.IndexOf(operation2.Operand2);
                        if (iOp2 == -1)
                        {
                            // If not, it's a data used only once
                            iOp2 = (short)dataList.IndexOf(operation2.Operand2);
                            // If not, something is wrong !
                            if (iOp2 == -1)
                                throw new Exception($"Index {i} ({operation2}) : Operand 2 {operation2.Operand2} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the register
                            if (!WillBeUsed(operation2.Operand2, topo, i))
                                registerList[iOp2] = null;
                            // Compute the KPU register index
                            iOp2 = (short)(-iOp2 - 1);
                        }
                    }
                    else
                        throw new NotSupportedException($"Operation {t} not supported.");
                }

                // If the operation is not the last one, the result is stored in a register. Otherwise, it's the final result.
                short iResult = (short)(i != topo.Count - 1 ? -Store(registerList, t) - 1 : OperationKPU.NoOperand);
                operations.Add(new OperationKPU(opCode, iResult, iOp1, iOp2));
            }

            datas.AddRange(dataList.Cast<TensorData<T>>());
            CacheSize = (byte)registerList.Count;
        }
    }
}
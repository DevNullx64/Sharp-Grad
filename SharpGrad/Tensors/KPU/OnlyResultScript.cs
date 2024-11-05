using SharpGrad.Tensors.KPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class OnlyResultScript<T> : KpuScrip<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        internal OnlyResultScript(ITensor<T> tensor)
        {
            var topo = tensor.DepthFirstSearch()
                .OrderBy(e => e.Value.Index)
                .Select(e => e.Value.Tensor)
                .ToList();

            List<Tensor<T>?> cacheList = [];

            for (int i = 0; i < topo.Count; i++)
            {
                var t = topo[i];
                OpCode opCode;
                OperandIndex indexOp1;
                OperandIndex indexOp2 = OperandIndex.Empty;

                if (t.Depth == 0)
                {
                    // Add the data tensor to the list of data tensors only if it is not already present
                    if (!operands.Contains(t))
                        operands.Add(t);


                    int uCount = UsageCount(t, topo, i);
                    // Skip caching if the data tensor is used only once
                    if (uCount == 1)
                        continue;

                    opCode = OpCode.Store;
                    indexOp1 = new(operands.IndexOf(t), OperandIndexSource.Operand);
                }
                else
                {
                    if (cacheList.Contains(t))
                        continue;

                    if (t.OperandCount == 1 && t is ITensorOperation1<T> operation1)
                    {
                        opCode = operation1.OpCode;

                        // OperationIndeces result or stored data should contains the operand
                        int iOp1 = cacheList.IndexOf(operation1.Operand);
                        if (iOp1 < 0)
                        {
                            // If not, it's a data used only once
                            iOp1 = operands.IndexOf(operation1.Operand);
                            // If not, something is wrong !
                            if (iOp1 < 0)
                                throw new Exception($"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");

                            indexOp1 = new(iOp1, OperandIndexSource.Operand);
                        }
                        else
                        {
                            // If the operand is not used anymore, free the cache
                            if (NextUse(operation1.Operand, topo, i) != -1)
                                cacheList[iOp1] = null;

                            indexOp1 = new(iOp1, OperandIndexSource.Cache);
                        }
                    }
                    else if (t.OperandCount == 2 && t is ITensorOperation2<T> operation2)
                    {
                        opCode = operation2.OpCode;

                        // OperationIndeces result or stored data should contains the first operand
                        int iOp1 = cacheList.IndexOf(operation2.Left);
                        if (iOp1 < 0)
                        {
                            // If not, it's a data used only once
                            iOp1 = operands.IndexOf(operation2.Left);
                            // If not, something is wrong !
                            if (iOp1 < 0)
                                throw new Exception($"Index {i} ({operation2}) : Operand 1 {operation2.Left} not found.");

                            indexOp1 = new(iOp1, OperandIndexSource.Operand);
                        }
                        else
                        {
                            // If the operand is not used anymore, free the cache
                            if (NextUse(operation2.Left, topo, i) != -1)
                                cacheList[iOp1] = null;

                            indexOp1 = new(iOp1, OperandIndexSource.Cache);
                        }

                        // OperationIndeces result or stored data should contains the second operand
                        int iOp2 = cacheList.IndexOf(operation2.Right);
                        if (iOp2 < 0)
                        {
                            // If not, it's a data used only once
                            iOp2 = operands.IndexOf(operation2.Right);
                            // If not, something is wrong !
                            if (iOp2 < 0)
                                throw new Exception($"Index {i} ({operation2}) : Operand 2 {operation2.Right} not found.");

                            indexOp2 = new(iOp2, OperandIndexSource.Operand);
                        }
                        else
                        {
                            // If the operand is not used anymore, free the cache
                            if (NextUse(operation2.Right, topo, i) != -1)
                                cacheList[iOp2] = null;

                            indexOp2 = new(iOp2, OperandIndexSource.Cache);
                        }
                    }
                    else
                        throw new NotSupportedException($"OperationIndeces {t} not supported.");
                }

                operations.Add(new OperationKPU(opCode,
                    // If the operation is not the last one, the result is stored in cache. Otherwise, it's the final result.
                    i < topo.Count
                        ? new(cacheList.Insert(t), ResultIndexSource.Cache)
                        : new(0, ResultIndexSource.Output),
                    indexOp1,
                    indexOp2
                    ));
            }

            CacheSize = (byte)cacheList.Count;
        }
    }
}
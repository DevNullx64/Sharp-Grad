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
                int iOp1;
                int iOp2 = int.MinValue;

                if (t.Depth == 0)
                {
                    // Add the data tensor to the list of data tensors only if it is not already present
                    if (!datas.Contains(t))
                        datas.Add(t);


                    int uCount = UsageCount(t, topo, i);
                    // Skip caching if the data tensor is used only once
                    if (uCount == 1)
                        continue;

                    opCode = OpCode.Store;
                    iOp1 = datas.IndexOf(t);
                }
                else
                {
                    if (cacheList.Contains(t))
                        continue;

                    if (t.OperandCount == 1 && t is ITensorOperation1<T> operation1)
                    {
                        opCode = operation1.OpCode;

                        // Operation result or stored data should contains the operand
                        iOp1 = cacheList.IndexOf(operation1.Operand);
                        if (iOp1 < 0)
                        {
                            // If not, it's a data used only once
                            iOp1 = datas.IndexOf(operation1.Operand);
                            // If not, something is wrong !
                            if (iOp1 < 0)
                                throw new Exception($"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the cache
                            if (NextUse(operation1.Operand, topo, i) != -1)
                                cacheList[iOp1] = null;
                            // Compute the KPU cache index
                            iOp1 = ~iOp1;
                        }
                    }
                    else if (t.OperandCount == 2 && t is ITensorOperation2<T> operation2)
                    {
                        opCode = operation2.OpCode;

                        // Operation result or stored data should contains the first operand
                        iOp1 = cacheList.IndexOf(operation2.Operand1);
                        if (iOp1 < 0)
                        {
                            // If not, it's a data used only once
                            iOp1 = datas.IndexOf(operation2.Operand1);
                            // If not, something is wrong !
                            if (iOp1 < 0)
                                throw new Exception($"Index {i} ({operation2}) : Operand 1 {operation2.Operand1} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the cache
                            if (NextUse(operation2.Operand1, topo, i) != -1)
                                cacheList[iOp1] = null;
                            // Compute the KPU cache index
                            iOp1 = ~iOp1;
                        }

                        // Operation result or stored data should contains the second operand
                        iOp2 = cacheList.IndexOf(operation2.Operand2);
                        if (iOp2 < 0)
                        {
                            // If not, it's a data used only once
                            iOp2 = datas.IndexOf(operation2.Operand2);
                            // If not, something is wrong !
                            if (iOp2 < 0)
                                throw new Exception($"Index {i} ({operation2}) : Operand 2 {operation2.Operand2} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the cache
                            if (NextUse(operation2.Operand2, topo, i) != -1)
                                cacheList[iOp2] = null;
                            // Compute the KPU cache index
                            iOp2 = ~iOp2;
                        }
                    }
                    else
                        throw new NotSupportedException($"Operation {t} not supported.");
                }

                operations.Add(new OperationKPU(opCode,
                    // If the operation is not the last one, the result is stored in cache. Otherwise, it's the final result.
                    i < topo.Count
                        ? new KPUIndex(cacheList.Insert(t), KPUIndexSource.Cache)
                        : KPUIndex.Empty,
                    iOp1 < 0
                        ? new KPUIndex(~iOp1, KPUIndexSource.Cache)
                        : new KPUIndex(iOp1, KPUIndexSource.Operation),
                    iOp2 == int.MinValue
                        ? KPUIndex.Empty
                        : iOp2 < 0
                            ? new KPUIndex(~iOp2, KPUIndexSource.Cache)
                            : new KPUIndex(iOp2, KPUIndexSource.Operation)
                    ));
            }

            CacheSize = (byte)cacheList.Count;
        }
    }
}
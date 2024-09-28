using SharpGrad.Tensors.KPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class KpuBackwardScript<T> : KpuScrip<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        private enum TreatmentState: byte
        {
            NotTreated = 0,
            LeftBranchTreated = 1,
            RightBranchTreated = 2,
            Treated = 3
        }
        protected readonly List<TensorData<T>> gradients = [];

        /// <summary>
        /// List of gradients to compute.
        /// </summary>
        public IReadOnlyList<TensorData<T>> Gradients => gradients;

        public KpuBackwardScript(Tensor<T> tensor)
        {
            // Topological sort of the graph that needs gradient computation
            List<Tensor<T>> topo = tensor.DepthFirstSearch()
                .OrderBy(e => e.Value.Index)
                .Select(e => e.Value.Tensor)
                .ToList();

            // List of data tensors that needs gradient computation
            gradients = topo.OfType<TensorData<T>>().ToList();

            TreatmentState[] treated = new TreatmentState[topo.Count];
            T[] gradients_ = new T[topo.Count];

            List<Tensor<T>?> cacheList = [];
            Stack<T> globalGradients = new();
            globalGradients.Push(T.One);

            // Follow the deep branch
            Tensor<T> current = topo[^1];
            while (current.OperandCount > 0)
            {
                if (current.OperandCount == 1)
                {
                    ITensorOperation1<T> operation1 = (ITensorOperation1<T>)current;
                    treated[topo.IndexOf(operation1.Operand)] = TreatmentState.Treated;
                    current = operation1.Operand;
                }
                else
                {
                    ITensorOperation2<T> operation2 = (ITensorOperation2<T>)current;
                    if(operation2.Operand1.Depth >= operation2.Operand2.Depth)
                    {
                        current = operation2.Operand1;
                        treated[topo.IndexOf(operation2.Operand2)] = TreatmentState.LeftBranchTreated;
                    } else
                    {
                        current = operation2.Operand2;
                        treated[topo.IndexOf(operation2.Operand1)] = TreatmentState.RightBranchTreated;
                    }
                }
            }

            // Follow the topological order
            for (int i = topo.Count - 1; i > 0; i--)
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
                        iOp1 = (short)cacheList.IndexOf(operation1.Operand);
                        if (iOp1 < 0)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)datas.IndexOf(operation1.Operand);
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
                        iOp1 = (short)cacheList.IndexOf(operation2.Operand1);
                        if (iOp1 < 0)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)datas.IndexOf(operation2.Operand1);
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
                        iOp2 = (short)cacheList.IndexOf(operation2.Operand2);
                        if (iOp2 < 0)
                        {
                            // If not, it's a data used only once
                            iOp2 = (short)datas.IndexOf(operation2.Operand2);
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
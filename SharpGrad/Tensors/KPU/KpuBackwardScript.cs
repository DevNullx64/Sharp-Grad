using ILGPU.Runtime.Cuda;
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
            List<Tensor<T>> topo = tensor.DepthFirstSearch(true)
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
            for (int i = topo.Count -1; i > 0; i--)
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
                    // Skip caching if the data tensor is used only once
                    if (uCount == 1)
                        continue;

                    opCode = OpCode.Store;
                    iOp1 = (short)datas.IndexOf(t);
                    iOp2 = OperationKPU.NoOperand;
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
                        if (iOp1 == -1)
                        {
                            // If not, it's a data used only once
                            iOp1 = (short)datas.IndexOf(operation1.Operand);
                            // If not, something is wrong !
                            if (iOp1 == -1)
                                throw new Exception($"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");
                        }
                        else
                        {
                            // If the operand is not used anymore, free the register
                            if (!WillBeUsed(operation1.Operand, topo, i))
                                cacheList[iOp1] = null;
                            // Compute the KPU register index
                            iOp1 = (short)(-iOp1 - 1);
                        }
                        iOp2 = OperationKPU.NoOperand;
                    }
                    else if (t.OperandCount == 2 && t is ITensorOperation2<T> operation2)
                    {
                        opCode = operation2.OpCode;

                        // Operation result or stored data should contains the first operand
                        iOp1 = (short)cacheList.IndexOf(operation2.Operand1);
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
                                cacheList[iOp1] = null;
                            // Compute the KPU register index
                            iOp1 = (short)(-iOp1 - 1);
                        }

                        // Operation result or stored data should contains the second operand
                        iOp2 = (short)cacheList.IndexOf(operation2.Operand2);
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
                                cacheList[iOp2] = null;
                            // Compute the KPU register index
                            iOp2 = (short)(-iOp2 - 1);
                        }
                    }
                    else
                        throw new NotSupportedException($"Operation {t} not supported.");
                }

                // If the operation is not the last one, the result is stored in a register. Otherwise, it's the final result.
                short iResult = (short)(i != topo.Count - 1 ? -Store(cacheList, t) - 1 : OperationKPU.NoOperand);
                operations.Add(new OperationKPU(opCode, iResult, iOp1, iOp2));
            }

            CacheSize = (byte)cacheList.Count;
        }
    }
}
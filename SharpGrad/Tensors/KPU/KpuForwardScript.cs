using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class KpuForwardScript<T> : KpuScrip<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        internal KpuForwardScript(Tensor<T> tensor)
        {
            var topo = tensor.DepthFirstSearch()
                .OrderBy(e => e.Value.Index)
                .Select(e => e.Value.Tensor)
                .ToList();

            for (int i = 0; i < topo.Count; i++)
            {
                var t = topo[i];
                OpCode opCode;
                int iOp1;
                int iOp2;
                int iResult = datas.IndexOf(t);

                // If 't' is not in the list, add it
                if (iResult < 0)
                {
                    // Add the data tensor to the list of tensors
                    iResult = datas.Count;
                    datas.Add(t);

                    switch (t.OperandCount)
                    {
                        // Data tensor
                        case 0:
                            continue;
                        
                        // One operand operation
                        case 1:
                            ITensorOperation1<T> operation1 = (ITensorOperation1<T>)t;
                            opCode = operation1.OpCode;

                            // Get the index of the operand
                            iOp1 = datas.IndexOf(operation1.Operand);
                            Debug.Assert(iOp1 >= 0, $"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");

                            iOp2 = OperationKPU.NoOperand;
                            break;

                        // Two operands operation
                        case 2:
                            ITensorOperation2<T> operation2 = (ITensorOperation2<T>)t;
                            opCode = operation2.OpCode;

                            // Operation result should contains the first operand
                            iOp1 = datas.IndexOf(operation2.Operand1);
                            Debug.Assert(iOp1 >= 0, $"Index {i} ({operation2}) : Operand 1 {operation2.Operand1} not found.");

                            // Operation result should contains the second operand
                            iOp2 = datas.IndexOf(operation2.Operand2);
                            Debug.Assert(iOp1 >= 0, $"Index {i} ({operation2}) : Operand 1 {operation2.Operand1} not found.");

                            break;

                        // Unsupported operation
                        default:
                            throw new NotSupportedException($"Operation {t} not supported.");
                    }
                }
                else
                {
                    switch (t.OperandCount)
                    {
                        case 0:
                            continue;

                        case 1:
                            ITensorOperation1<T> operation1 = (ITensorOperation1<T>)t;
                            opCode = operation1.OpCode;

                            iOp1 = datas.IndexOf(operation1.Operand);
                            Debug.Assert(iOp1 >= 0, $"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");

                            iOp2 = OperationKPU.NoOperand;
                            break;

                        case 2:
                            ITensorOperation2<T> operation2 = (ITensorOperation2<T>)t;
                            opCode = operation2.OpCode;

                            iOp1 = datas.IndexOf(operation2.Operand1);
                            Debug.Assert(iOp1 >= 0, $"Index {i} ({operation2}) : Operand 1 {operation2.Operand1} not found.");

                            iOp2 = datas.IndexOf(operation2.Operand2);
                            Debug.Assert(iOp2 >= 0, $"Index {i} ({operation2}) : Operand 2 {operation2.Operand2} not found.");
                            break;

                        default:
                            throw new NotSupportedException($"Operation {t} not supported.");
                    }
                }
                checked
                {
                    operations.Add(new OperationKPU(opCode, (short)iResult, (short)iOp1, (short)iOp2));
                }
            }
            CacheSize = 0;
        }
    }
}
using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors.KPU
{
    public class AllResultScript<T> : KpuScrip<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        internal AllResultScript(Tensor<T> tensor)
        {
            var topo = tensor.DepthFirstSearch()
                .OrderBy(e => e.Value.Index)
                .Select(e => e.Value.Tensor)
                .ToList();

            for (int i = 0; i < topo.Count; i++)
            {
                var t = topo[i];
                OpCode opCode;
                OperandIndex indexLeft;
                OperandIndex indexRight = OperandIndex.Empty;
                ResultIndex indexResult;

                // If 't' is not in the list, add it
                int iResult = operands.IndexOf(t);
                if (iResult < 0)
                {
                    // Add the data tensor to the list of tensors
                    indexResult = new(operands.Count, ResultIndexSource.Cache);
                    operands.Add(t); 

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
                            int iLeft = operands.IndexOf(operation1.Operand);
                            Debug.Assert(iLeft >= 0, $"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");

                            indexLeft = new(iLeft, OperandIndexSource.Operand);
                            break;

                        // Two tensors operation
                        case 2:
                            ITensorOperation2<T> operation2 = (ITensorOperation2<T>)t;
                            opCode = operation2.OpCode;

                            // Operation result should contains the first operand
                            iLeft = operands.IndexOf(operation2.Left);
                            Debug.Assert(iLeft >= 0, $"Index {i} ({operation2}) : Operand 1 {operation2.Left} not found.");
                            
                            // Operation result should contains the second operand
                            int iRight = operands.IndexOf(operation2.Right);
                            Debug.Assert(iRight >= 0, $"Index {i} ({operation2}) : Operand 1 {operation2.Right} not found.");

                            indexLeft = new(iLeft, OperandIndexSource.Operand);
                            indexRight = new(iRight, OperandIndexSource.Operand);
                            break;

                        // Unsupported operation
                        default:
                            throw new NotSupportedException($"Operation {t} not supported.");
                    }
                }
                else
                {
                    indexResult = new(iResult, ResultIndexSource.Cache);

                    switch (t.OperandCount)
                    {
                        case 0:
                            continue;

                        case 1:
                            ITensorOperation1<T> operation1 = (ITensorOperation1<T>)t;
                            opCode = operation1.OpCode;

                            int iLeft = operands.IndexOf(operation1.Operand);
                            Debug.Assert(iLeft >= 0, $"Index {i} ({operation1}) : Operand 1 {operation1.Operand} not found.");

                            indexLeft = new(iLeft, OperandIndexSource.Operand);
                            break;

                        case 2:
                            ITensorOperation2<T> operation2 = (ITensorOperation2<T>)t;
                            opCode = operation2.OpCode;

                            iLeft = operands.IndexOf(operation2.Left);
                            Debug.Assert(iLeft >= 0, $"Index {i} ({operation2}) : Operand 1 {operation2.Left} not found.");

                            int iRight = operands.IndexOf(operation2.Right);
                            Debug.Assert(iRight >= 0, $"Index {i} ({operation2}) : Operand 2 {operation2.Right} not found.");

                            indexLeft = new(iLeft, OperandIndexSource.Operand);
                            indexRight = new(iRight, OperandIndexSource.Operand);
                            break;

                        default:
                            throw new NotSupportedException($"Operation {t} not supported.");
                    }
                }
                checked
                {
                    operations.Add(new OperationKPU(opCode,
                        indexResult,
                        indexLeft,
                        indexRight));
                }
            }
            CacheSize = 0;
        }
    }
}
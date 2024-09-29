using ILGPU.Runtime;
using ILGPU;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using SharpGrad.Tensors.Operators;
using System.Runtime.CompilerServices;
using System.Runtime;
using ILGPU.Runtime.Cuda;
using System.Data.SqlTypes;
using System.Data;
using System.Net.Http.Headers;
using System.Diagnostics;
using SharpGrad.Tensors.KPU;

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit
    {
        private static void ForwardKernel<T>(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<T, Stride2D.DenseY> tensors)
            where T : unmanaged, INumber<T>
        {
            for (int i = 0; i < ops.Length; i++)
            {
                OperationKPU op = ops[i];

                T op1 = tensors[op.LeftOperand.Value, idx];
                tensors[op.IndexResult.Value, idx] = op.RightOperand.IsEmpty 
                    ? Exec(op.OpCode, op1)
                    : Exec(op.OpCode, op1, tensors[op.RightOperand.Value, idx]);
            }
        }

        public void Forward<T>(Tensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            if (tensor is ITensorOperation<T> tensorOperation)
            {
                AllResultScript<T> script = tensor.ForwardScript;
                using MemoryBuffer2D<T, Stride2D.DenseY> tensors = To2D(script.Datas.Select(e => e.View));
                AcceleratorBuffer<OperationKPU> ops = MMU.GetBuffer(script.ToArray());
                var func = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OperationKPU>, ArrayView2D<T, Stride2D.DenseY>>(ForwardKernel);
                func(new Index1D((int)tensors.Extent.Y), ops.AcceleratorData.View, tensors.View);
                Synchronize();
                var results = GetRows(tensors);

                for (int i = 0; i < results.Count; i++)
                    script.Datas[i].Buffer = DefaultKPU.MMU.GetBuffer(results[i]);
            }
        }
    }
}

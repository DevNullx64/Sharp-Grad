using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit
    {
        #region KPU Script execution
        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="operation">Operation to perform</param>
        /// <param name="operand1">Left operand</param>
        /// <param name="result">Output of the operation</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T Exec<T>(OpCode operation, T operand1, T operand2)
            where T : unmanaged, INumber<T>
        {
            return operation switch
            {
                //OpCode.Reset => T.Zero,
                OpCode.Store => operand1,
                OpCode.Add => AddOp<T>.Exec(operand1, operand2),
                OpCode.Sub => SubOp<T>.Exec(operand1, operand2),
                OpCode.Mul => MulOp<T>.Exec(operand1, operand2),
                OpCode.Div => DivOp<T>.Exec(operand1, operand2),
                // OpCode.Pow => PowOp<T>.Exec(operand1, operand2),
                OpCode.Neg => NegOp<T>.Exec(operand1),
                //OpCode.Log => LogOp<T>.Exec(operand1),
                //OpCode.Exp => ExpOp<T>.Exec(operand1),
                _ => T.Zero,
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T Exec<T>(OpCode operation, T operand1)
            where T : unmanaged, INumber<T>
        {
            return operation switch
            {
                //OpCode.Reset => T.Zero,
                OpCode.Neg => NegOp<T>.Exec(operand1),
                //OpCode.Log => LogOp<T>.Exec(operand1),
                //OpCode.Exp => ExpOp<T>.Exec(operand1),
                _ => T.Zero,
            };
        }

        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="idx">GPU Index</param>
        /// <param name="ops">Operations to perform</param>
        /// <param name="tensors">Tensors to operate on</param>
        /// <param name="registerCount">Number of registers to use</param>
        private static T ExecKernel<T>(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<T, Stride2D.DenseY> tensors, T[] cache)
            where T : unmanaged, INumber<T>
        {
            T result = T.Zero;

            for (int i = 0; i < ops.Length; i++)
            {
                OperationKPU op = ops[i];

                short v1 = op.IndexOperand1.Value;
                T op1 = op.IndexOperand1.Source == KPUIndexSource.Cache
                    ? cache[v1]
                    : tensors[v1, idx];

                result = op.IndexOperand2.IsEmpty
                    ? Exec(op.OpCode, op1)
                    : Exec(op.OpCode, op1, tensors[op.IndexOperand2.Value, idx]);

                if (i < ops.Length - 1)
                {
                    short v2 = op.IndexResult.Value;
                    if (op.IndexResult.Source == KPUIndexSource.Cache)
                        cache[v2] = result;
                    else
                        tensors[v2, idx] = result;
                }
            }

            return result;
        }

        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="idx">GPU Index</param>
        /// <param name="ops">Operations to perform</param>
        /// <param name="tensors">Tensors to operate on</param>
        /// <param name="registerCount">Number of registers to use</param>
        private static void ExecKernel<T>(
            Index1D idx,
            ArrayView<OperationKPU> ops,
            ArrayView2D<T, Stride2D.DenseY> tensors,
            ArrayView1D<T, Stride1D.Dense> result,
            SpecializedValue<short> registerCount)
            where T : unmanaged, INumber<T>
        {
            T[] register = new T[registerCount];
            result[idx] = ExecKernel(idx, ops, tensors, register);
        }
        #endregion

        #region Manage 2D tensors
        private static void GetRowKernel<T>(Index1D idx, ArrayView2D<T, Stride2D.DenseY> tensors, ArrayView1D<T, Stride1D.Dense> result, int row)
            where T : unmanaged
        {
            //Debug.Assert(result.Length == tensors.Extent.Y, $"Invalid {nameof(result)} row length {result.Length} for {nameof(tensors)} shape {tensors.Extent.Y}");
            result[idx] = tensors[row, idx];
        }

        private void GetRow<T>(MemoryBuffer2D<T, Stride2D.DenseY> tensor, int row, MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged
        {
            var GetRowFnc = Accelerator.LoadAutoGroupedKernel<Index1D, ArrayView2D<T, Stride2D.DenseY>, ArrayView1D<T, Stride1D.Dense>, int>(GetRowKernel);
            GetRowFnc(Accelerator.DefaultStream, new Index1D((int)result.Length), tensor.View, result.View, row);
        }

        private MemoryBuffer1D<T, Stride1D.Dense> GetRow<T>(MemoryBuffer2D<T, Stride2D.DenseY> tensor, int row)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var result = Accelerator.Allocate1D<T, Stride1D.Dense>(tensor.IntExtent.Y, new Stride1D.Dense());
            GetRow(tensor, row, result);
            return result;
        }
        private List<MemoryBuffer1D<T, Stride1D.Dense>> GetRows<T>(MemoryBuffer2D<T, Stride2D.DenseY> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            List<MemoryBuffer1D<T, Stride1D.Dense>> result = [];
            for (int i = 0; i < tensor.IntExtent.X; i++)
                result.Add(GetRow(tensor, i));
            return result;
        }

        private static void SetRowKernel<T>(Index1D idx, ArrayView1D<T, Stride1D.Dense> tensor, ArrayView2D<T, Stride2D.DenseY> results, int row)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            => results[row, idx] = tensor[idx];

        private MemoryBuffer2D<T, Stride2D.DenseY> To2D<T>(IEnumerable<ArrayView1D<T, Stride1D.Dense>> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var length = tensor.First().Length;

            MemoryBuffer2D<T, Stride2D.DenseY> result = Accelerator.Allocate2DDenseY<T>(new(tensor.Count(), length));
            var func = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView2D<T, Stride2D.DenseY>, int>(SetRowKernel);

            int i = 0;
            foreach (var t in tensor)
            {
                if (t.Length != length)
                    throw new InvalidOperationException($"Invalid data length {t.Length} for shape {length}");
                func(new Index1D((int)length), t, result.View, i++);
            }
            Synchronize();
            return result;
        }
        #endregion

        public void Compute<T>(Tensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            if (tensor is ITensorOperation<T> tensorOperation)
            {
                OnlyResultScript<T> script = tensor.ExecScript;
                using MemoryBuffer2D<T, Stride2D.DenseY> tensors = To2D(script.Datas.Select(e => e.View));
                AcceleratorBuffer<OperationKPU> ops = MMU.GetBuffer(script.ToArray());
                AcceleratorBuffer<T> resultBuffer = MMU.GetBuffer<T>(tensors.Extent.Y);
                var func = Accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, // GPU Index in result tensor
                    ArrayView<OperationKPU>, // Operations to perform
                    ArrayView2D<T, Stride2D.DenseY>, // Tensors to operate on
                    ArrayView1D<T, Stride1D.Dense>, // Output tensor
                    SpecializedValue<short>> // Number of cached results
                    (ExecKernel);
                func(
                    new Index1D((int)resultBuffer.AcceleratorData.Length), 
                    ops.AcceleratorData.View,
                    tensors.View,
                    resultBuffer.AcceleratorData.View, 
                    new SpecializedValue<short>(script.CacheSize));
                Synchronize();
                tensor.Buffer = resultBuffer;
            }
        }

        /// <summary>
        /// Compute the result of the tensor operation and reduce the result tensor in the specified dimension.
        /// </summary>
        /// <typeparam name="T">Type of the tensor</typeparam>
        /// <typeparam name="TOp">Type of the reduction operation</typeparam>
        /// <param name="tensor">Tensor to compute</param>
        /// <param name="dim">Dimension to reduce</param>
        /// <returns></returns>
        public TensorData<T> Compute<T, TOp>(Tensor<T> tensor, Index? dim = null)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            where TOp : IExecOperation<T, T, T>
        {
            dim ??= ^1;
            byte dim_ = (byte)(dim.Value.IsFromEnd ? tensor.Shape.Count - dim.Value.Value : dim.Value.Value);
            if (tensor is ITensorOperation<T> tensorOperation)
            {
                OnlyResultScript<T> script = tensor.ExecScript;
                using MemoryBuffer2D<T, Stride2D.DenseY> tensors = To2D(script.Datas.Select(e => e.View));
                AcceleratorBuffer<OperationKPU> ops = MMU.GetBuffer(script.ToArray());
                AcceleratorBuffer<int> shapeBuffer = MMU.GetBuffer((int[])tensor.Shape);
                Shape outputShape = tensor.Shape.SetDim(dim_, (tensor.Shape[dim_] + ReduceKernelElementsCount - 1) / ReduceKernelElementsCount);
                AcceleratorBuffer<T> resultBuffer = MMU.GetBuffer<T>(outputShape.Length);

                ByteArgs args = new(script.CacheSize, (byte)tensor.Shape.Count, dim_, 32);
                var func = Accelerator.LoadAutoGroupedStreamKernel <
                    Index1D, // GPU Index in result tensor
                    ArrayView< OperationKPU >, // Operations to perform
                    ArrayView2D<T, Stride2D.DenseY>, // Tensors to operate on
                    ArrayView1D<T, Stride1D.Dense>, // Output tensor
                    ArrayView1D< int, Stride1D.Dense > , // Shape of the input tensor
                    SpecializedValue<ByteArgs> > // Number of elements to reduce
                    (ReduceKernel<T, TOp>);
                func(
                    new Index1D((int)resultBuffer.AcceleratorData.Length),
                    ops.AcceleratorData.View,
                    tensors.View, resultBuffer.AcceleratorData.View,
                    shapeBuffer.AcceleratorData.View,
                    new SpecializedValue<ByteArgs>(args));
                Synchronize();
                return new TensorData<T>("Output", outputShape, resultBuffer);
            }
            else
                return (TensorData<T>)tensor;
        }
    }
}

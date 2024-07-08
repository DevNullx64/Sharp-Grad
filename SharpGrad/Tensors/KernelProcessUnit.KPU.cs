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

namespace SharpGrad.Tensors
{
    // A : Alpha   : α : Α
    // B : Beta    : β : Β
    // G : Gamma   : γ : Γ
    // D : Delta   : δ : Δ
    // E : Epsilon : ε : Ε
    // Z : Zeta    : ζ : Ζ
    // H : Eta     : η : Η
    // T : Theta   : θ : Θ
    // I : Iota    : ι : Ι
    // K : Kappa   : κ : Κ
    // L : Lambda  : λ : Λ
    // M : Mu      : μ : Μ
    // N : Nu      : ν : Ν
    // X : Xi      : ξ : Ξ
    // O : Omicron : ο : Ο
    // P : Pi      : π : Π
    // R : Rho     : ρ : Ρ
    // S : Sigma   : σ : Σ
    // T : Tau     : τ : Τ
    // U : Upsilon : υ : Υ
    // F : Phi     : φ : Φ
    // C : Chi     : χ : Χ
    // Y : Psi     : ψ : Ψ
    // W : Omega   : ω : Ω

    // Matematical symbols
    // ∑  : Summation
    // ∏  : Product
    // ∫  : Integral
    // ∂  : Partial
    // ∇ : Nabla
    // √  : Square root
    // ∛  : Cube root
    // ∜  : Fourth root
    // ∝ : Proportional to
    // ∞  : Infinity
    // ∟  : Right angle
    // ∠ : Angle
    // ∡  : Measured angle
    // ∢  : Spherical angle
    // ∣  : Divides
    // ∤  : Does not divide
    // ∥ : Parallel to
    // ∦  : Not parallel to
    // ∧ : Logical and
    // ∨ : Logical or
    // ∩  : Intersection
    // ∪ : Union

    public partial class KernelProcessUnit
    {
        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="operation">Operation to perform</param>
        /// <param name="operand1">Left operand</param>
        /// <param name="result">Result of the operation</param>
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

        /// <summary>
        /// Kernel Processing Unit
        /// </summary>
        /// <param name="idx">GPU Index</param>
        /// <param name="ops">Operations to perform</param>
        /// <param name="tensors">Tensors to operate on</param>
        /// <param name="registerCount">Number of registers to use</param>
        private static T ExecKernel_<T>(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<T, Stride2D.DenseY> tensors, T[] register)
            where T : unmanaged, INumber<T>
        {
            T result = T.Zero;

            for (int i = 0; i < ops.Length; i++)
            {
                OperationKPU op = ops[i];

                result = Exec(op.OpCode,
                    op.IndexOperand1 < 0
                        ? register[~op.IndexOperand1]
                        : tensors[op.IndexOperand1, idx],
                    op.IndexOperand2 < 0
                        ? register[~op.IndexOperand2]
                        : tensors[op.IndexOperand2, idx]);

                if (i < ops.Length - 1)
                {
                    if (op.IndexResult >= 0)
                        tensors[op.IndexResult, idx] = result;
                    else
                        register[~op.IndexResult] = result;
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
        private static void ExecKernel<T>(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<T, Stride2D.DenseY> tensors, ArrayView1D<T, Stride1D.Dense> result, SpecializedValue<short> registerCount)
            where T : unmanaged, INumber<T>
        {
            T[] register = new T[registerCount];
            result[idx] = ExecKernel_(idx, ops, tensors, register);
        }

        private static void ExecKernel<T, TOp>(
            Index1D idx, 
            ArrayView<OperationKPU> ops, 
            ArrayView2D<T, Stride2D.DenseY> tensors, 
            SpecializedValue<short> registerCount)
            where T : unmanaged, INumber<T>
            where TOp : IExecutor2<T, T, T>
        {
            T[] register = new T[registerCount];

            T result = ExecKernel_(idx, ops, tensors, register);
            OperationKPU op = ops[ops.Length - 1];
            if (op.IndexResult >= 0)
                tensors[op.IndexResult, idx] = result;
            else
                register[~op.IndexResult] = result;
        }

        private static void GetRowKernel<T>(Index1D idx, ArrayView2D<T, Stride2D.DenseY> tensors, ArrayView1D<T, Stride1D.Dense> result, int row)
            where T : unmanaged
            => result[idx] = tensors[row, idx];
        private void GetRow<T>(MemoryBuffer2D<T, Stride2D.DenseY> tensor, int row, MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged
        {
            Action<AcceleratorStream, Index1D, ArrayView2D<T, Stride2D.DenseY>, ArrayView1D<T, Stride1D.Dense>, int> GetRowFnc
                = Accelerator.LoadAutoGroupedKernel<Index1D, ArrayView2D<T, Stride2D.DenseY>, ArrayView1D<T, Stride1D.Dense>, int>(GetRowKernel);
            GetRowFnc(Accelerator.DefaultStream, new Index1D((int)result.Length), tensor.View, result.View, row);
        }

        private MemoryBuffer1D<T, Stride1D.Dense> GetRow<T>(MemoryBuffer2D<T, Stride2D.DenseY> tensor, int row)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var result = Accelerator.Allocate1D<T, Stride1D.Dense>(tensor.IntExtent.Y, new Stride1D.Dense());
            GetRow(tensor, row, result);
            return result;
        }


        private static void SetRowKernel<T>(Index1D idx, ArrayView1D<T, Stride1D.Dense> tensor, ArrayView2D<T, Stride2D.DenseY> results, SpecializedValue<int> row)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            => results[row, idx] = tensor[idx];
        private MemoryBuffer2D<T, Stride2D.DenseY> To2D<T>(IEnumerable<ArrayView1D<T, Stride1D.Dense>> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var length = tensor.First().Length;

            MemoryBuffer2D<T, Stride2D.DenseY> result = Accelerator.Allocate2DDenseY<T>(new(tensor.Count(), length));
            var func = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView2D<T, Stride2D.DenseY>, SpecializedValue<int>>(SetRowKernel);

            int i = 0;
            foreach (var t in tensor)
            {
                if (t.Length != length)
                    throw new InvalidOperationException($"Invalid data length {t.Length} for shape {length}");
                func(new Index1D((int)length), t, result.View, new SpecializedValue<int>(i++));
            }
            Synchronize();
            return result;
        }

        public TensorData<T> Compute<T>(Tensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            if (tensor is ITensorOperation<T> tensorOperation)
            {
                KpuScript<T> script = GetKpuScript(tensor);
                using MemoryBuffer2D<T, Stride2D.DenseY> tensors = To2D(script.Datas.Select(e => e.View));
                AcceleratorBuffer<OperationKPU> ops = MMU.GetBuffer(script.ToArray());
                AcceleratorBuffer<T> resultBuffer = MMU.GetBuffer<T>(tensors.Extent.Y);
                var func = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OperationKPU>, ArrayView2D<T, Stride2D.DenseY>, ArrayView1D<T, Stride1D.Dense>, SpecializedValue<short>>(ExecKernel);
                func(new Index1D((int)resultBuffer.AcceleratorData.Length), ops.AcceleratorData.View, tensors.View, resultBuffer.AcceleratorData.View, new SpecializedValue<short>(script.RegistersCount));
                Synchronize();
                return new TensorData<T>("Result", tensor.Shape, resultBuffer);
            }
            else
                return (TensorData<T>)tensor;
        }

        private int ReduceKernelElementsCount = 32;
        private static void ReduceKernel<T, TOp>(
            Index1D idxDestination,
            ArrayView1D<T, Stride1D.Dense> source,
            ArrayView1D<int, Stride1D.Dense> sourceShape,
            ArrayView1D<T, Stride1D.Dense> destination,
            int dim,
            int count,
            SpecializedValue<int> dims)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            where TOp : IExecutor2<T, T, T>
        {
            // Duplicate the tensor shape. Except for the dimension to reduce, where the size is divided by the count.
            int[] destinationShape = new int[dims];
            for (int i = 0; i < dims; i++)
                destinationShape[i] = (i == dim) ? (sourceShape[i] + count - 1) / count : sourceShape[i];

            // Get the indices of input and output tensors.
            int[] indicesDestination = Shape.IndicesFrom(destinationShape, idxDestination);
            int[] indicesSource = new int[dims];
            for(int i = 0; i < dims; i++)
                indicesSource[i] = (i == dim) ? indicesDestination[i] * count : indicesDestination[i];

            // Compute the amount of elements that can be reduced.
            int cMax = indicesSource[dim] + count;
            if (cMax > sourceShape[dim])
                cMax = sourceShape[dim];

            // Compute the reduction of the two first elements.
            int iSource = Shape.GetFlattenIndices(sourceShape, indicesSource);
            indicesSource[dim]++;
            int iSource2 = Shape.GetFlattenIndices(sourceShape, indicesSource);
            indicesSource[dim]++;
            T acc = TOp.Exec(source[iSource], source[iSource2]);

            // Reduce the elements.
            for (; indicesSource[dim] < cMax; indicesSource[dim]++)
            {
                iSource2 = Shape.GetFlattenIndices(sourceShape, indicesSource);
                acc = TOp.Exec(acc, source[iSource2]);
            }

            // Store the result.
            destination[idxDestination] = acc;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ReduceKernel<T, TOp>(
            Index1D idxResults,
            ArrayView1D<T, Stride1D.Dense> from,
            ArrayView1D<int, Stride1D.Dense> fromShape,
            ArrayView1D<T, Stride1D.Dense> results,
            int dim)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            where TOp : IExecutor2<T, T, T>
        { ReduceKernel<T, TOp>(idxResults, from, fromShape, results, dim, 32, new SpecializedValue<int>((int)fromShape.Length)); }

        public TensorData<T> Reduce<T, TOp>(Tensor<T> tensor, Index? dim = null)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            where TOp : IExecutor2<T, T, T>
        {
            // If dim is not specified, reduce the last dimension.
            int dim_ = (dim is null)
                ? tensor.Shape.Count - 1 // [^1] by default
                : (dim.Value.IsFromEnd)
                    ? tensor.Shape.Count - dim.Value.Value
                    : dim.Value.Value;

            // Compute the tensor if it is not already computed.
            // Remember that a tensor without operations is already computed.
            if (tensor.OperandCount != 0)
                tensor = Compute(tensor);

            // If the dimension to reduce is already 1, return the tensor.
            if (tensor.Shape[dim_] == 1)
                return (TensorData<T>)tensor;

            // Compute the shape of the result tensor.
            var sourceShape = tensor.Shape;
            int resultingSize = (tensor.Shape[dim_] + ReduceKernelElementsCount - 1) / ReduceKernelElementsCount;
            var destinationShape = tensor.Shape.SetDim(dim_, resultingSize);

            var shapeGpu = MMU.GetBuffer((int[])sourceShape);
            var destinationGpu = MMU.GetBuffer<T>(destinationShape.Length);

            var fnc = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, int, int, SpecializedValue<int>>(ReduceKernel<T, TOp>);

            if (tensor is TensorData<T> tensorData)
            {
                fnc(
                    new Index1D((int)destinationShape.Length),
                    tensorData.View,
                    MMU.GetBuffer((int[])tensorData.Shape).AcceleratorData.View,
                    destinationGpu.AcceleratorData.View,
                    dim_,
                    ReduceKernelElementsCount,
                    new SpecializedValue<int>(tensorData.Shape.Count));

                // Reduce the tensor until it has only one element in the dimension to reduce.
                AcceleratorBuffer<T>? sourceGpu = null;
                while (resultingSize > 1)
                {
                    (sourceGpu, destinationGpu) = (destinationGpu, sourceGpu);

                    sourceShape = destinationShape;

                    resultingSize = (resultingSize + ReduceKernelElementsCount - 1) / ReduceKernelElementsCount;
                    destinationShape = sourceShape.SetDim(dim_, resultingSize);

                    destinationGpu ??= MMU.GetBuffer<T>(destinationShape.Length); // Allocate the destination buffer the first time.

                    fnc(
                        new Index1D((int)destinationShape.Length),
                        sourceGpu.AcceleratorData.View,
                        MMU.GetBuffer((int[])sourceShape).AcceleratorData.View,
                        destinationGpu.AcceleratorData.View,
                        dim_,
                        ReduceKernelElementsCount,
                        new SpecializedValue<int>(sourceShape.Count));
                }
                if (sourceGpu is not null)
                {
                    sourceGpu.Dispose();
                    sourceGpu = destinationGpu;
                    destinationGpu = MMU.GetBuffer<T>(destinationShape.Length);
                    destinationGpu.AcceleratorData.CopyFrom(sourceGpu);
                    sourceGpu.Dispose();
                }
                return new TensorData<T>($"{tensor.Name} {{result}}", new Shape(destinationShape), destinationGpu);
            }
            else
                throw new Exception($"This should not happen.");
        }
    }
}

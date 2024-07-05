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
        private static void ExecKernel<T>(Index1D idx, ArrayView<OperationKPU> ops, ArrayView2D<T, Stride2D.DenseY> tensors, SpecializedValue<short> registerCount)
            where T : unmanaged, INumber<T>
        {
            T[] register = new T[registerCount];

            for (int i = 0; i < ops.Length; i++)
            {
                OperationKPU op = ops[i];
                var operand1 = op.IndexOperand1 < 0
                    ? register[-op.IndexOperand1 - 1]
                    : tensors[op.IndexOperand1, idx];
                var operand2 = op.IndexOperand2 < 0
                    ? register[-op.IndexOperand2 - 1]
                    : tensors[op.IndexOperand2, idx];
                if (op.IndexResult >= 0)
                    tensors[op.IndexResult, idx] = Exec(op.OpCode, operand1, operand2);
                else
                    register[-op.IndexResult - 1] = Exec(op.OpCode, operand1, operand2);
            }
        }

        private void Exec<T>(Index1D idx, MemoryBuffer1D<OperationKPU, Stride1D.Dense> ops, MemoryBuffer2D<T, Stride2D.DenseY> tensors, short registerCount)
            where T : unmanaged, INumber<T>
        {
            var func = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OperationKPU>, ArrayView2D<T, Stride2D.DenseY>, SpecializedValue<short>>(ExecKernel);
            func(idx, ops.View, tensors.View, new SpecializedValue<short>(registerCount));
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
            //foreach (var t in tensor)
            //{
            //    // offload 1D tensor.
            //}
            return result;
        }

        internal TensorData<T> Exec<T>(IEnumerable<OperationKPU> operations, IEnumerable<TensorData<T>> datas, int registryCount)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var lastOperation = operations.Last();
            short resultRow = lastOperation.IndexResult;
            if (lastOperation.IndexResult < 0) checked
                {
                    // The result is stored in a register. Need to allocate a new tensor to store the result.
                    resultRow = (short)datas.Count();
                    TensorData<T> result = ("Result", new Shape((int)datas.First().Length));
                    datas = datas.Append(result);
                    // Add a store operation to store the result in the result tensor.
                    operations = operations.Append(new OperationKPU(OpCode.Store, resultRow, (short)datas.Count()));
                }
            using MemoryBuffer2D<T, Stride2D.DenseY> tensors = To2D(datas.Select(e => e.View));

            AcceleratorBuffer<OperationKPU> ops = GetBuffer(operations.ToArray());

            var func = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OperationKPU>, ArrayView2D<T, Stride2D.DenseY>, SpecializedValue<short>>(ExecKernel);
            func(new Index1D(operations.Count()), ops.AcceleratorData.View, tensors.View, new SpecializedValue<short>((short)registryCount));
            Synchronize();

            var resultMemory = GetRow(tensors, resultRow);
            AcceleratorBuffer<T> resultBuffer = ((ILowLevelMemoryManager)this).GetBuffer(resultMemory);
            return new TensorData<T>("Result", new Shape((int)resultMemory.Length), resultBuffer);
        }

        public TensorData<T> Exec<T>(Tensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            if (tensor is ITensorOperation<T> tensorOperation)
            {
                KpuScript<T> script = GetKpuScript(tensor);

                using MemoryBuffer2D<T, Stride2D.DenseY> tensors = To2D(script.Datas.Select(e => e.View));
                AcceleratorBuffer<OperationKPU> ops = GetBuffer(script.ToArray());
                var func = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OperationKPU>, ArrayView2D<T, Stride2D.DenseY>, SpecializedValue<short>>(ExecKernel);
                func(new Index1D((int)tensors.Extent.Y), ops.AcceleratorData.View, tensors.View, new SpecializedValue<short>(script.RegistersCount));
                Synchronize();

                var resultMemory = GetRow(tensors, 0);
                Synchronize();
                AcceleratorBuffer<T> resultBuffer = ((ILowLevelMemoryManager)this).GetBuffer(resultMemory);
                return new TensorData<T>("Result", new Shape((int)resultMemory.Length), resultBuffer);
            }
            else
                return (TensorData<T>)tensor;
        }

        private static void ReduceStage1Kernel<T, TOp>(
            Index1D idx, 
            ArrayView1D<T, Stride1D.Dense> tensor, 
            ArrayView1D<T, Stride1D.Dense> results, 
            ArrayView1D<int, Stride1D.Dense> shape, 
            int dim, 
            SpecializedValue<int> count)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            where TOp : IExecutor2<T, T, T>
        {
            int[] intShape = [shape.IntLength];
            for(int i = 0; i < shape.Length; i++)
                intShape[i] = shape[i];

            int[] indices_to = Shape.IndicesFrom(intShape, idx);
            int[] indices_from = [indices_to.Length];
            for (int i = 0; i < indices_to.Length; i++)
                indices_from[i] = (i == dim) ? indices_to[i] * count : indices_to[i];

            T acc = TOp.Exec(tensor[Shape.GetFlattenIndices(indices_from, intShape)], tensor[Shape.GetFlattenIndices(indices_from, intShape)]);
            int cMax = indices_from[dim] + count <= intShape[dim] ? indices_from[dim] + count : intShape[dim];
            for (; indices_from[dim] < cMax; indices_from[dim]++)
                acc = TOp.Exec(acc, tensor[Shape.GetFlattenIndices(indices_from, intShape)]);
            results[idx] = acc;
        }

        public TensorData<T> Reduce<T, TOp>(Tensor<T> tensor, Index? dim = null)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            where TOp : IExecutor2<T, T, T>
        {
            dim ??= new Index(1, true);

            SpecializedValue<int> dims = (dim.Value.IsFromEnd)
                ? new SpecializedValue<int>(tensor.Shape.Count - dim.Value.Value)
                : new SpecializedValue<int>(dim.Value.Value);

            var fnc = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int, SpecializedValue<int>>(ReduceStage1Kernel<T, TOp>);

            var shape = new int[tensor.Shape.Count];
            for (int i = 0; i < tensor.Shape.Count; i++)
                shape[i] = tensor.Shape[i].Size;

            var result = Accelerator.Allocate1D<T, Stride1D.Dense>(shape[dims], new Stride1D.Dense());
            AcceleratorBuffer<int> shapeGpu = GetBuffer(shape);

            if (tensor is TensorData<T> tensorData)
            {
                fnc(new Index1D((int)result.Length), tensorData.View, result.View, shapeGpu.AcceleratorData.View, dims, new SpecializedValue<int>(32));
                return new TensorData<T>("Result", new Shape(shape[dims]), ((ILowLevelMemoryManager)this).GetBuffer(result));
            }
            else
                throw new NotImplementedException();
        }
    }
}

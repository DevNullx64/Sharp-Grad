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

        private static void GetRowKernel<T>(LongIndex1D idx, ArrayView2D<T, Stride2D.DenseY> tensors, ArrayView1D<T, Stride1D.Dense> result, SpecializedValue<int> row)
            where T : unmanaged
            => result[idx] = tensors[row, idx];
        private void GetRow<T>(MemoryBuffer2D<T, Stride2D.DenseY> tensor, int row, MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged
        {
            Action<AcceleratorStream, LongIndex1D, ArrayView2D<T, Stride2D.DenseY>, ArrayView1D<T, Stride1D.Dense>, SpecializedValue<int>> GetRowFnc 
                = Accelerator.LoadAutoGroupedKernel<LongIndex1D, ArrayView2D<T, Stride2D.DenseY>, ArrayView1D<T, Stride1D.Dense>, SpecializedValue<int>>(GetRowKernel);
            GetRowFnc(Accelerator.DefaultStream, new LongIndex1D(tensor.IntExtent.Y), tensor.View, result.View, new SpecializedValue<int>(row));
        }

        private MemoryBuffer1D<T, Stride1D.Dense> GetRow<T>(MemoryBuffer2D<T, Stride2D.DenseY> tensor, int row)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var result = Accelerator.Allocate1D<T, Stride1D.Dense>(tensor.IntExtent.X, new Stride1D.Dense());
            GetRow(tensor, row, result);
            return result;
        }


        private static void SetRowKernel<T>(LongIndex1D idx, ArrayView1D<T, Stride1D.Dense> tensors, ArrayView2D<T, Stride2D.DenseY> result, SpecializedValue<int> row)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            => result[row, idx] = tensors[idx];
        private MemoryBuffer2D<T, Stride2D.DenseY> To2D<T>(IEnumerable<ArrayView1D<T, Stride1D.Dense>> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            long length = tensor.First().Length;

            MemoryBuffer2D<T, Stride2D.DenseY> result = Accelerator.Allocate2DDenseY<T>(new(tensor.Count(), length));
            var func = Accelerator.LoadAutoGroupedStreamKernel<LongIndex1D, ArrayView1D<T, Stride1D.Dense>, ArrayView2D<T, Stride2D.DenseY>, SpecializedValue<int>>(SetRowKernel);

            int i = 0;
            foreach (var t in tensor)
            {
                if (t.Length != length)
                    throw new InvalidOperationException($"Invalid data length {t.Length} for shape {length}");
                func(new LongIndex1D(length), t, result.View, new SpecializedValue<int>(i++));
            }
            Synchronize();
            //foreach (var t in tensor)
            //{
            //    // offload 1D tensor.
            //}
            return result;
        }

        internal Tensor<T> Exec<T>(IEnumerable<OperationKPU> operations, IEnumerable<DataTensor<T>> datas, int registryCount)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {

            var lastOperation = operations.Last();
            short resultRow = lastOperation.IndexResult;
            if (lastOperation.IndexResult < 0) checked
            {
                // The result is stored in a register. Need to allocate a new tensor to store the result.
                resultRow = (short)datas.Count();
                DataTensor<T> result = ("Result", new Shape((int)datas.First().Length));
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
            return new DataTensor<T>("Result", new Shape((int)resultMemory.Length), resultBuffer);
        }

        /// <summary>
        /// Get the KPU index of the tensor. If the tensor is not stored in a register, store it in a register.
        /// </summary>
        /// <typeparam name="T">The type of the tensor</typeparam>
        /// <param name="registers">The KPU context</param>
        /// <param name="operand">The tensor to get the KPU index</param>
        /// <param name="iOperand">The KPU index of the tensor</param>
        /// <returns>False if the tensor is already stored in a register. True if a new register is used as iOperand and need an additional store operation.</returns>
        /// <exception cref="InvalidOperationException">If the tensor require the result of another operation that wasn't stored in a register yet.</exception>
        private static bool GetOperandIndex<T>(KPUContext<T> registers, Tensor<T> operand, out short iOperand)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            short? iOp = registers.GetKpuIndex(operand);
            if (iOp.HasValue)
            {
                iOperand = iOp.Value;
                return false;
            }
            else
            {
                // The operans is a 'multiple usage' input tensor. Need to store it in a register before using it.
                if (operand.Depth == 0)
                {
                    iOperand = registers.Store(operand);
                    return true;
                }
                else
                    throw new InvalidOperationException($"This operation require the result of another operation. The operand {operand} is not stored in a register.");
            }
        }

        public Tensor<T> Exec<T>(Tensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            if (tensor is ITensorOperation<T> tensorOperation)
            {
                var topo = new List<Tensor<T>>();
                var visited = new HashSet<Tensor<T>>();
                tensorOperation.DepthFirstSearch(topo, visited);

                KPUContext<T> registers = new(topo);

                // Compile / Convert to Kpu operations
                List<OperationKPU> operations = [];
                short result = OperationKPU.Empty;
                foreach (var operation in topo)
                {
                    // Ignore input tensors. Input is treated when treating operands of an operation.
                    if (operation.Depth > 0)
                    {
                        OpCode opCode;
                        // Get the result register
                        result = registers.GetKpuIndex(operation)
                            ?? registers.Store(operation); // No allocation needed. It will assigned at runtime as result.
                        short iOp1 = OperationKPU.Empty;
                        short iOp2 = OperationKPU.Empty;
                        if (operation is ITensorOperation1<T> op1)
                        {
                            opCode = op1.OpCode;

                            if (GetOperandIndex(registers, op1.Operand1, out iOp1))
                                operations.Add(new OperationKPU(OpCode.Store, iOp1, registers.IndexInDatas(op1.Operand1)));

                            registers.Use(op1.Operand1);
                        }
                        else if (operation is ITensorOperation2<T> op2)
                        {
                            opCode = op2.OpCode;

                            if (GetOperandIndex(registers, op2.Operand1, out iOp1))
                                operations.Add(new OperationKPU(OpCode.Store, iOp1, registers.IndexInDatas(op2.Operand1)));
                            if (GetOperandIndex(registers, op2.Operand2, out iOp2))
                                operations.Add(new OperationKPU(OpCode.Store, iOp2, registers.IndexInDatas(op2.Operand2)));

                            registers.Use(op2.Operand1);
                            registers.Use(op2.Operand2);
                        }
                        else if (operation is ITensorReduce<T> opR)
                        {
                            throw new NotImplementedException();
                        }
                        else
                        {
                            throw new InvalidOperationException($"Invalid operation {operation}");
                        }

                        operations.Add(new OperationKPU(opCode, result, iOp1, iOp2));
                    }
                }
                return Exec(operations, registers.Datas.Select(e => (DataTensor<T>)e.Tensor), registers.KPURegisters.Count);
            }
            else
                return tensor;
        }
    }
}

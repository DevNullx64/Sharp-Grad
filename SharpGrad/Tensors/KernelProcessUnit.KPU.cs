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
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            return operation switch
            {
                OpCode.Reset => T.Zero,
                OpCode.Store => operand1,
                OpCode.Add => AddOp<T>.Exec(operand1, operand2),
                OpCode.Sub => SubOp<T>.Exec(operand1, operand2),
                OpCode.Mul => MulOp<T>.Exec(operand1, operand2),
                OpCode.Div => DivOp<T>.Exec(operand1, operand2),
                OpCode.Pow => PowOp<T>.Exec(operand1, operand2),
                OpCode.Neg => NegOp<T>.Exec(operand1),
                OpCode.Log => LogOp<T>.Exec(operand1),
                OpCode.Exp => ExpOp<T>.Exec(operand1),
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
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
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

        private static void GetRowKernel<T>(LongIndex1D idx, ArrayView2D<T, Stride2D.DenseY> tensors, ArrayView1D<T, Stride1D.Dense> result, SpecializedValue<int> row)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
            => result[idx] = tensors[row, idx];
        private MemoryBuffer1D<T, Stride1D.Dense> GetRow<T>(MemoryBuffer2D<T, Stride2D.DenseY> tensor, int row)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var result = Accelerator.Allocate1D<T, Stride1D.Dense>(tensor.IntExtent.X, new Stride1D.Dense());
            GetRowKernel(new LongIndex1D(tensor.IntExtent.X), tensor.View, result.View, new SpecializedValue<int>(row));
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

        public MemoryBuffer1D<T, Stride1D.Dense> Exec<T>(OperationKPU[] operations)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            var needAccumulator = operations.Min(e => Math.Min(e.IndexOperand1, e.IndexOperand2));
            if (needAccumulator < 0)
                needAccumulator = (short)-needAccumulator;
            else
                needAccumulator = 0;

            var ops = Accelerator.Allocate1D<OperationKPU, Stride1D.Dense>(operations.Length, new Stride1D.Dense());
            ops.CopyFromCPU(operations);

            var tensors = Accelerator.Allocate2DDenseY<T>(new(operations.Max(e => e.IndexOperand1), operations.Length));

            // TODO : Copy input

            //ExecKernel(new Index1D(tensors.IntExtent.Y), ops.View, tensors.View, new SpecializedValue<short>(needAccumulator));

            // TODO : Return tensor
            return tensors[0];
        }

        public readonly struct KPUContext<T>(IEnumerable<Tensor<T>> tensors)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            public struct KPURegister(Tensor<T> tensor, int ttl)
            {
                public Tensor<T>? Value = tensor;
                private int TTL = ttl;

                public void Use()
                {
                    if (Value is null || TTL <= 0)
                        throw new InvalidOperationException($"Try to use an empty register");
                    if (--TTL <= 0)
                    {
                        Value = null;
                    }
                }
            }

            // contains all input tensors and number of use
            private readonly List<(Tensor<T> Tensor, short TTL)> inputs = tensors
                .Where(e => e.Depth == 0)
                .GroupBy(e => e)
                .Select(e => (e.Key, (short)e.Count()))
                .ToList();
            public readonly IReadOnlyList<(Tensor<T> Tensor, short TTL)> Inputs => inputs;

            /// <summary>
            /// Get the index of the input tensor
            /// </summary>
            /// <param name="tensor">Tensor to get the index</param>
            /// <returns>Index of the input tensor. -1 if not found</returns>
            public short IndexOfInputs(Tensor<T> tensor) => (short)inputs.IndexOf(inputs.First(e => e.Tensor == tensor));

            // contains all operations, and number of use
            private readonly List<(Tensor<T> Tensor, short TTL)> operations = tensors
                .Where(e => e.Depth > 0)
                .GroupBy(e => e)
                .Select(e => (e.Key, (short)e.Count()))
                .ToList();

            public readonly IReadOnlyList<(Tensor<T> Tensor, short TTL)> Operations => operations;

            private readonly List<KPURegister> kpuRegisters = [];
            public IReadOnlyList<KPURegister> KPURegisters => kpuRegisters;

            public short GetTTL(Tensor<T> tensor)
            {
                foreach (var (t, ttl) in tensor.Depth == 0 ? inputs : operations)
                    if (t == tensor)
                        return ttl;
                throw new InvalidOperationException($"Tensor {tensor} not found for this graph");
            }

            /// <summary>
            /// Get the register index of the tensor
            /// </summary>
            /// <param name="tensor">Tensor to get the register index</param>
            /// <param name="firstEmpty">Index of the first empty register</param>
            /// <returns>Register index or -1 if not found</returns>
            public short GetRegisterIndex(Tensor<T> tensor)
            {
                for (short i = 0; i < kpuRegisters.Count; i++)
                    if (tensor == kpuRegisters[i].Value)
                        return i;
                return -1;
            }

            /// <summary>
            /// Get the KPU index of the tensor.
            /// </summary>
            /// <param name="tensor">Tensor to get the KPU index</param>
            /// <returns>If the tensor was found in the regitry list or input list, return the  KPU index. Otherwise, return null.</returns>
            /// <remarks>A KPU index is negative if the tensor is stored in a register. The realy index in the register list is (-index - 1). Otherwise, the index is the index of the input tensor.</remarks>
            public short? GetKpuIndex(Tensor<T> tensor)
            {
                short regIndex = GetRegisterIndex(tensor);
                if (regIndex >= 0)
                {
                    kpuRegisters[regIndex].Use();
                    return (short)(-regIndex - 1);
                }
                else if (tensor.Depth == 0)
                {
                    int i;
                    int ttl = 0;

                    // Get the ttl of the tensor
                    for (i = 0; i < inputs.Count; i++)
                    {
                        (Tensor<T> t, ttl) = inputs[i];
                        if (t == tensor)
                            break;
                    }

                    // Check if we need a regiter or not
                    if (ttl == 1)
                        return (short)i;
                }
                return null;
            }

            public short Store(Tensor<T> tensor)
            {
                short ttl = GetTTL(tensor);
                if (tensor.Depth == 0 && ttl == 1)
                    throw new InvalidOperationException($"Tensor {tensor} is an input tensor used only once. No need to store it in a register.");

                kpuRegisters.Add(new KPURegister(tensor, ttl));
                return (short)-kpuRegisters.Count;
            }

            public void Use(Tensor<T> tensor)
            {
                short regIndex = GetRegisterIndex(tensor);
                if (regIndex < 0)
                    throw new InvalidOperationException($"Tensor {tensor} not found in the register list");

                kpuRegisters[regIndex].Use();

                if (tensor is ITensorOperation1<T> operation1)
                {
                    Use(operation1.Operand1);
                }
                else if (tensor is ITensorOperation2<T> operation2)
                {
                    Use(operation2.Operand1);
                    Use(operation2.Operand2);
                }
            }

            public void UseAt(short index) => kpuRegisters[index].Use();
            public void UseAtKpuIndex(short index) => UseAt((short)(-index - 1));
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
                // For each operation, convert it to a KPU operation.
                // All temporary tensors are stored in registers.
                // All input tensors are stored in memory.
                // All output tensors are stored in memory.
                List<OperationKPU> operations = [];
                foreach (var operation in topo)
                {
                    // Ignore input tensors. Input is treated when treat the operands of an operation.
                    if (operation.Depth > 0)
                    {
                        OpCode opCode;
                        // Get the result register
                        var result = registers.GetKpuIndex(operation)
                            ?? registers.Store(operation);
                        short? iOp1 = OperationKPU.Empty;
                        short? iOp2 = OperationKPU.Empty;
                        if (operation is ITensorOperation1<T> op1)
                        {
                            opCode = op1.OpCode;
                            // Get the operand register / 'unique usage' input tensor index
                            iOp1 = registers.GetKpuIndex(op1.Operand1);
                            if (!iOp1.HasValue)
                            {
                                // The operans is a 'multiple usage' input tensor. Need to store it in a register before using it.
                                if (op1.Operand1.Depth == 0)
                                {
                                    iOp1 = registers.Store(op1.Operand1);
                                    operations.Add(new OperationKPU(OpCode.Store, registers.Store(op1.Operand1), iOp1.Value));
                                    registers.Use(op1.Operand1);
                                }
                                else
                                    throw new InvalidOperationException($"This operation require the result of another operation. The operand {op1.Operand1} is not stored in a register.");
                            }
                            registers.Use(op1.Operand1);
                        }
                        else if (operation is ITensorOperation2<T> op2)
                        {
                            throw new NotImplementedException();
                        }
                        else if (operation is ITensorReduce<T> opR)
                        {
                            throw new NotImplementedException();
                        }
                        else
                        {
                            throw new InvalidOperationException($"Invalid operation {operation}");
                        }

                        operations.Add(new OperationKPU(opCode, result, iOp1.Value, iOp2.HasValue ? iOp2.Value : OperationKPU.Empty));
                    }
                }

            }
            return tensor;
        }
    }
}

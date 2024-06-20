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

        public Tensor<T> Exec<T>(Tensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            if (tensor is ITensorOperation<T> op)
            {
                var topo = new List<Tensor<T>>();
                var visited = new HashSet<Tensor<T>>();
                op.DepthFirstSearch(topo, visited);

                // Get all input (Tensor<T>.Depth == 0) and is number of use
                var inputs = topo
                    .Where(e => e.Depth == 0)
                    .GroupBy(e => e)
                    .ToDictionary(e => e.Key, e => e.Count());

                // Get all output (Tensor<T>.Depth == max) and is number of use
                var usesOutput = topo
                    .Where(e => e.Depth == topo.Max(e => e.Depth))
                    .GroupBy(e => e)
                    .ToDictionary(e => e.Key, e => e.Count());
            }
        }
    }
}
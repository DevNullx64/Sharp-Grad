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
        /// <param name="operand">Left operand</param>
        /// <param name="result">Result of the operation</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Exec<T>(OpCode operation, ref T result, T operand)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            result = operation switch
            {
                OpCode.Reset => T.Zero,
                OpCode.Store => operand,
                OpCode.Add => AddOp<T>.Exec(result, operand),
                OpCode.Sub => SubOp<T>.Exec(result, operand),
                OpCode.Mul => MulOp<T>.Exec(result, operand),
                OpCode.Div => DivOp<T>.Exec(result, operand),
                OpCode.Pow => PowOp<T>.Exec(result, operand),
                OpCode.Neg => NegOp<T>.Exec(operand),
                OpCode.Log => LogOp<T>.Exec(operand),
                OpCode.Exp => ExpOp<T>.Exec(operand),
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
                if (op.Index1 >= 0)
                    Exec(op.OpCode, ref tensors[op.Index1, idx], op.Index2 >= 0 ? tensors[op.Index2, idx] : register[-op.Index2 - 1]);
                else
                    Exec(op.OpCode, ref register[-op.Index1 - 1], op.Index2 >= 0 ? tensors[op.Index2, idx] : register[-op.Index2 - 1]);
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
            var needAccumulator = operations.Min(e => Math.Min(e.Index1, e.Index2));
            if (needAccumulator < 0)
                needAccumulator = (short)-needAccumulator;
            else
                needAccumulator = 0;

            var ops = Accelerator.Allocate1D<OperationKPU, Stride1D.Dense>(operations.Length, new Stride1D.Dense());
            ops.CopyFromCPU(operations);

            var tensors = Accelerator.Allocate2DDenseY<T>(new(operations.Max(e => e.Index1), operations.Length));

            // TODO : Copy input

            //ExecKernel(new Index1D(tensors.IntExtent.Y), ops.View, tensors.View, new SpecializedValue<short>(needAccumulator));

            // TODO : Return tensor
            return tensors[0];
        }

        public class RegisteryContext : List<int>
        {
            public int FirstFreeRegister => Find(e => e == -1);
        }

        public Tensor<T> Exec<T>(Tensor<T> tensor)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
        {
            if (tensor is ITensorOperation<T> op)
            {
                var topo = new List<ITensorOperation<T>>();
                var visited = new Dictionary<Tensor<T>, int>();
                var leaf = new Dictionary<Tensor<T>, int>();
                op.DepthFirstSearch(topo, visited, leaf);

                int requiredInputs = leaf.Count;

                // Get every opeartion, from the bagining, an stop at first false, with the same shape
                int s = 0;
                Shape current = topo[s].Shape;
                int e = s + 1;
                while (e < topo.Count && topo[e].Shape == current)
                    e++;

                // Get needed operands
                HashSet<DataTensor<T>> operands = [];
                for (int i = s; i < e; i++)
                    if (topo[i] is ITensorOperation1<T> executor)
                    {
                        if (executor.Operand1 is DataTensor<T> data)
                            operands.Add(data);
                    }
                    else if (topo[i] is ITensorOperation2<T> executor2)
                    {
                        if (executor2.Operand1 is DataTensor<T> data1)
                            operands.Add(data1);
                        if (executor2.Operand2 is DataTensor<T> data2)
                            operands.Add(data2);
                    }


                // Use ILGPU to pack operands
                MemoryBuffer2D<T, Stride2D.DenseY> operandsMatrix = To2D(operands.Select(e => e.view));

                // Create list  of required registers
                RegisteryContext registry = [];
                for (int i = s; i < e; i++)
                {
                    if (topo[i] is IExecutor1<T, T> executor)
                    {
                        int p = registry.FirstFreeRegister;
                        if (p == -1)
                        {
                            registry.Add(i);
                        }
                    }
                    else if (topo[i] is IExecutor2<T, T, T> executor2)
                    {

                    }
                }


            }

        }
    }
}
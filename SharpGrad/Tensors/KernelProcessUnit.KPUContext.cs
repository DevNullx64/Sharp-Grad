using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{


    public partial class KernelProcessUnit
    {
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
            private readonly List<(Tensor<T> Tensor, short TTL)> datas = tensors
                .Where(e => e.Depth == 0)
                .GroupBy(e => e)
                .Select(e => (e.Key, (short)e.Count()))
                .ToList();
            public readonly IReadOnlyList<(Tensor<T> Tensor, short TTL)> Datas => datas;

            /// <summary>
            /// Get the index of the input tensor
            /// </summary>
            /// <param name="tensor">Tensor to get the index</param>
            /// <returns>Index of the input tensor. -1 if not found</returns>
            public short IndexInDatas(Tensor<T> tensor) => (short)datas.IndexOf(datas.First(e => e.Tensor == tensor));

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
                foreach (var (t, ttl) in tensor.Depth == 0 ? datas : operations)
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
                    for (i = 0; i < datas.Count; i++)
                    {
                        (Tensor<T> t, ttl) = datas[i];
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
    }
}

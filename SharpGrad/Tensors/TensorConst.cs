using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public class TensorConst<T> : Tensor<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>, IExponentialFunctions<T>, ILogarithmicFunctions<T>
    {
        public override long Depth => 0;
        public override int OperandCount => 0;

        public override bool NeedsGradient => false;

        protected TensorConst(Shape shape, AcceleratorBuffer<T> buffer, string? name = null)
            : base(shape, buffer, name)
        { }

        public TensorConst(Shape shape, T[] data, string? name = null)
            : this(shape, KernelProcessUnit.DefaultKPU.MMU.GetBuffer(data), name)
        { }

        public override bool Equals(ITensor? other)
            => other is TensorConst<T> tensor && Buffer == tensor.Buffer;

        public override bool Equals(object? obj) => obj is TensorData<T> tensor && Equals(tensor);

        public override string ToString() => $"{Name}{Shape}";

        internal override void DepthFirstSearch(Dictionary<Tensor<T>, DfsNode<T>> topoSort, DepthFirstSearchOption needsGradientOnly = DepthFirstSearchOption.None)
        {
            if (topoSort.TryGetValue(this, out DfsNode<T>? node))
                node.UsageCount++;
            else
            {
                if (needsGradientOnly.HasFlag(DepthFirstSearchOption.AllGradient) || NeedsGradient)
                    topoSort.Add(this, new(this, topoSort.Count, 1));
            }
        }

        public override void Backward()
        {
            throw new NotSupportedException($"TensorData<{typeof(T).Name}> does not support backward operation");
        }

    }
}

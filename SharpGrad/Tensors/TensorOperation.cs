using ILGPU;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Formats.Tar;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class TensorOperationOne<T, TOp>(Tensor<T> left) : TensorBase<T>(left.Shape)
        where T : unmanaged, IFloatingPoint<T>
        where TOp : struct, IBackwardOne<T>
    {
        public readonly TensorBase<T> LeftOperand = left;

        private readonly DeviceBuffer<T> data = new(left.Length);
        internal override DeviceBuffer<T> Data
        {
            get
            {
                if (data.IsEmpty)
                {
                    ExecGpu(TOp.ApplyGpu, LeftOperand.Data.DeviceData, data.DeviceData);
                }
                return data;
            }
        }

        protected T GetData(int[] indices) => data.CPUData[shape.GetFlattenedIndex(indices)];

        public override T this[params int[] indices] {
            get => GetData(indices);
            set => throw new NotImplementedException($"Cannot set value to {GetType().Name}");
        }
    }

    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class NegOperation<T>(Tensor<T> left) : TensorOperationOne<T, NegOp<T>>(left)
        where T : unmanaged, IFloatingPoint<T>
    { }

    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class ReLUOperation<T>(Tensor<T> left) : TensorOperationOne<T, ReLUOp<T>>(left)
        where T : unmanaged, IFloatingPoint<T>
    { }
}

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

        private readonly AcceleratorBuffer<T> data = new(left.Length);
        internal override AcceleratorBuffer<T> Data
        {
            get
            {
                if (data.IsEmpty)
                {
                    ExecGpu(TOp.ApplyGpu, LeftOperand.Data.AcceleratorData, data.AcceleratorData);
                }
                return data;
            }
        }

        public override T this[params int[] indices] {
            get => data.CPUData[shape.GetFlattenedIndex(indices)];
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


    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class TensorOperationTwo<T, Top>(Tensor<T> left, Tensor<T> right) : TensorBase<T>(left.Shape)
        where T : unmanaged, IFloatingPoint<T>
        where Top : struct, IBackwardTwo<T>
    {
        public readonly TensorBase<T> LeftOperand = left.Length == right.Length ? left : throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");
        public readonly TensorBase<T> RightOperand = right;

        private readonly AcceleratorBuffer<T> data = new(left.Length);
        internal override AcceleratorBuffer<T> Data
        {
            get
            {
                if (data.IsEmpty)
                {
                    ExecGpu(Top.ApplyGpu, LeftOperand.Data.AcceleratorData, RightOperand.Data.AcceleratorData, data.AcceleratorData);
                }
                return data;
            }
        }

        protected T GetData(int[] indices) => data.CPUData[shape.GetFlattenedIndex(indices)];

        public override T this[params int[] indices]
        {
            get => GetData(indices);
            set => throw new NotImplementedException($"Cannot set value to {GetType().Name}");
        }
    }
}

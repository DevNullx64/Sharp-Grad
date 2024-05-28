using ILGPU;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Formats.Tar;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class TensorOperations<T, TOp, TGrad>(Tensor<T, TGrad> left) : TensorBase<T, TGrad>(left.Shape)
        where T : unmanaged, INumber<T> 
        where TOp : struct, IBackwardOne<T, TGrad>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public readonly TensorBase<T, TGrad> LeftOperand = left;

        private readonly AcceleratorBuffer<T> data = new(left.Length);
        internal override AcceleratorBuffer<T> Data
        {
            get
            {
                if (data.IsEmpty)
                {
                    ExecAccelerator(TOp.ApplyGpu, LeftOperand.Data.AcceleratorData, data.AcceleratorData);
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
    internal class NegOperation<T, TGrad>(Tensor<T, TGrad> left) : TensorOperations<T, NegOp<T, TGrad>, TGrad>(left)
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }

    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class ReLUOperation<T, TGrad>(Tensor<T, TGrad> left) : TensorOperations<T, ReLUOp<T, TGrad>, TGrad>(left)
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    { }


    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class TensorOperationTwo<T, Top, TGrad>(TensorBase<T, TGrad> left, TensorBase<T, TGrad> right) : TensorBase<T, TGrad>(left.Shape)
        where T : unmanaged, INumber<T>
        where Top : struct, IBackwardTwo<T, TGrad>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public readonly TensorBase<T, TGrad> LeftOperand = left.Length == right.Length ? left : throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");
        public readonly TensorBase<T, TGrad> RightOperand = right;

        private readonly AcceleratorBuffer<T> data = new(left.Length);
        internal override AcceleratorBuffer<T> Data
        {
            get
            {
                if (data.IsEmpty)
                {
                    ExecAccelerator(Top.ApplyGpu, LeftOperand.Data.AcceleratorData, RightOperand.Data.AcceleratorData, data.AcceleratorData);
                }
                return data;
            }
        }

        public override T this[params int[] indices]
        {
            get => Top.ApplyCpu(LeftOperand[indices], RightOperand[indices]);
            set => throw new NotImplementedException($"Cannot set value to {GetType().Name}");
        }
    }

    /*
    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class TensorOperationTwo<T, Top>(TensorBase<T> left, TensorBase<T> right) : TensorBase<T>(left.Shape)
        where T : unmanaged, INumber<T>
        where Top : struct, IApplyOpTwo<T>
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
                    ExecAccelerator(Top.ApplyGpu, LeftOperand.Data.AcceleratorData, RightOperand.Data.AcceleratorData, data.AcceleratorData);
                }
                return data;
            }
        }

        public override T this[params int[] indices]
        {
            get => Top.ApplyCpu(LeftOperand[indices], RightOperand[indices]);
            set => throw new NotImplementedException($"Cannot set value to {GetType().Name}");
        }
    */
}

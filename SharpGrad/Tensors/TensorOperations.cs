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
    internal class TensorOperations<T, TOp, TGrad>(Tensor<T, TGrad> left, bool isGrad = false) : Tensor<T, TGrad>(left.Shape, isGrad)
        where T : unmanaged, INumber<T>
        where TOp : struct, IBackwardOne<T, TGrad>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public readonly Tensor<T, TGrad> LeftOperand = left;

        private readonly AcceleratorBuffer<T>? data = isGrad ? new(left.Length) : null;
        internal override AcceleratorBuffer<T> Data
        {
            get
            {
                if (data.IsEmpty)
                {
                    ExecAccelerator(TOp.ApplyAccelerator, LeftOperand.Data.AcceleratorData, data.AcceleratorData);
                }
                return data;
            }
        }

        public override bool IsBackward => true;

        public override T this[params int[] indices]
        {
            get => IsGradients
                ? Data.CPUData[shape.GetFlattenedIndex(indices)]
                : TOp.ApplyCpu(LeftOperand[indices]);
            set => throw new NotImplementedException($"Cannot set value to {GetType().Name}");
        }

        public override void Dispose()
        {
            data?.Dispose();
            base.Dispose();
        }
    }


    [SuppressMessage("Usage", "CA2260:Use the correct type parameter", Justification = "Take into account in the architecture. A bad T type should be impossible.")]
    internal class TensorOperationTwo<T, TOp, TGrad>(Tensor<T, TGrad> left, Tensor<T, TGrad> right, bool isGrad = false) : Tensor<T, TGrad>(left.Shape, isGrad)
        where T : unmanaged, INumber<T>
        where TOp : struct, IBackwardTwo<T, TGrad>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public readonly Tensor<T, TGrad> LeftOperand = left.Length == right.Length ? left : throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");
        public readonly Tensor<T, TGrad> RightOperand = right;

        private readonly AcceleratorBuffer<T>? data = isGrad ? new(left.Length) : null;
        internal override AcceleratorBuffer<T> Data
        {
            get
            {
                if (data.IsEmpty)
                {
                    ExecAccelerator(TOp.ApplyAccelerator, LeftOperand.Data.AcceleratorData, RightOperand.Data.AcceleratorData, data.AcceleratorData);
                }
                return data;
            }
        }

        public override bool IsBackward => true;

        public override T this[params int[] indices]
        {
            get => IsGradients 
                ? Data.CPUData[shape.GetFlattenedIndex(indices)] 
                : TOp.ApplyCpu(LeftOperand[indices], RightOperand[indices]);
            set => throw new NotImplementedException($"Cannot set value to {GetType().Name}");
        }

        public override void Dispose()
        {
            data?.Dispose();
            base.Dispose();
        }
    }
}

using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    public abstract class Tensor<T, TGrad>(Shape shape, bool isGrad) : ITensor<Tensor<T, TGrad>, T, TGrad>, IDisposable
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static readonly Tensor<T, TGrad> Empty = new DataTensor<T, TGrad>();

        protected readonly Shape shape = shape;
        public Shape Shape => shape;

        internal abstract AcceleratorBuffer<T> Data { get; }

        internal AcceleratorBuffer<T>? gradients = isGrad ? new(shape.Size) : null;
        public void AddGradient(AcceleratorBuffer<T> gradient)
        {
            if (Length != gradient.Length)
                throw new ArgumentException($"Expected length {Length}, got {gradient.Length}");
            if (IsGradients)
#pragma warning disable CS8604 // Checked by IsGrad
                Acc.Exec<T>(AddOp<T, float>.ApplyAccelerator, gradients, gradient, gradients);
#pragma warning restore CS8604
        }

        public void ResetGradients()
        {
            if (IsGradients)
                gradients?.MemSetToZero();
        }

        public bool IsGradients
        {
            get => gradients is not null;
            set
            {
                if(value != IsGradients)
                {
                    if(value)
                        gradients = new(shape.Size);
                    else
                    {
                        gradients?.Dispose();
                        gradients = null;
                    }
                }
            }
        }
        public virtual bool IsBackward { get; } = false;

        public readonly long Length = shape.Size;

        public abstract T this[params int[] indices] { get; set; }

        public static void DynAccelerator(
            OpCode[] operations,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            Action<Index1D, ArrayView<OpCode>, ArrayView<T>, ArrayView<T>, ArrayView<T>> loadedKernel =
                Acc.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OpCode>, ArrayView<T>, ArrayView<T>, ArrayView<T>>(KernelProcessUnit<T, TGrad>.Dynamic);
            loadedKernel(left.IntExtent, Acc.Accelerator.Allocate1D(operations).View, left.View, right.View, result.View);
            Acc.Accelerator.Synchronize();
        }

        public static void DynAccelerator(
            OpCode[] operations,
            Tensor<T, TGrad> left, Tensor<T, TGrad> right, Tensor<T, TGrad> result)
            => DynAccelerator(operations, left.Data.AcceleratorData, right.Data.AcceleratorData, result.Data.AcceleratorData);

        public virtual void Dispose()
        {
            if (gradients is not null)
            {
                gradients.Dispose();
                gradients = null;
            }
            GC.SuppressFinalize(this);
        }

        public static Tensor<T, TGrad> operator +(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => new TensorOperationTwo<T, AddOp<T, TGrad>, TGrad>(left, right);
        public static Tensor<T, TGrad> operator -(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => new TensorOperationTwo<T, SubOp<T, TGrad>, TGrad>(left, right);
        public static Tensor<T, TGrad> operator *(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => new TensorOperationTwo<T, MulOp<T, TGrad>, TGrad>(left, right);
        public static Tensor<T, TGrad> operator /(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => new TensorOperationTwo<T, DivOp<T, TGrad>, TGrad>(left, right);

    }
}
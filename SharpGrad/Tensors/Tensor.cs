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
        public void AddGradient(DataTensor<T, TGrad> gradient)
        {
            if (gradient.shape != shape)
                throw new ArgumentException($"Expected gradient shape {shape}, got {gradient.shape}");
            if (IsGradients)
#pragma warning disable CS8604 // Checked by IsGrad
                ExecAccelerator(AddOp<T, TGrad>.ApplyAccelerator, gradients, gradient.Data, gradients);
#pragma warning restore CS8604
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


        public static void ExecAccelerator(
            Action<Index1D, ArrayView<T>, ArrayView<T>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            Action<Index1D, ArrayView<T>, ArrayView<T>> loadedKernel = Tensors.Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, result.View);
            Tensors.Accelerator.Synchronize();
        }

        public static void ExecAccelerator(
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> loadedKernel = Tensors.Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, right.View, result.View);
            Tensors.Accelerator.Synchronize();
        }


        public static void ExecAccelerator(
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> func,
            Tensor<T, TGrad> left, Tensor<T, TGrad> right, Tensor<T, TGrad> result)
            => ExecAccelerator(func, left.Data.AcceleratorData, right.Data.AcceleratorData, result.Data.AcceleratorData);

        public static DataTensor<T, TGrad> ExecAccelerator(
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> func,
            Tensor<T, TGrad> left, Tensor<T, TGrad> right)
        {
            if (left.shape != right.shape)
                throw new ArgumentException($"Expected shapes {left.shape}, got {right.shape}");
            var result = new DataTensor<T, TGrad>(left.shape);
            ExecAccelerator(func, left, right, result);
            return result;
        }

        public static void ExecAccelerator(
            OpCode[] operations,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            if(left.Length != right.Length || left.Length != result.Length)
                throw new ArgumentException($"Length mismatch: {nameof(left)}:{left.Length}, {nameof(right)}:{right.Length}, {nameof(result)}:{result.Length}");
            ExecAccelerator(operations, left, right, result);
        }

        public static void DynAccelerator(
            OpCode[] operations,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            Action<Index1D, ArrayView<OpCode>, ArrayView<T>, ArrayView<T>, ArrayView<T>> loadedKernel =
                Tensors.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OpCode>, ArrayView<T>, ArrayView<T>, ArrayView<T>>(KernelProcessUnit<T, TGrad>.Dynamic);
            loadedKernel(left.IntExtent, Tensors.Accelerator.Allocate1D(operations).View, left.View, right.View, result.View);
            Tensors.Accelerator.Synchronize();
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

        public void ResetGradients()
        {
            if (IsGradients)
                gradients?.MemSetToZero();
        }

        public static Tensor<T, TGrad> operator +(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => new TensorOperationTwo<T, AddOp<T, TGrad>, TGrad>(left, right);
        public static Tensor<T, TGrad> operator -(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => new TensorOperationTwo<T, SubOp<T, TGrad>, TGrad>(left, right);
        public static Tensor<T, TGrad> operator *(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => new TensorOperationTwo<T, MulOp<T, TGrad>, TGrad>(left, right);
        public static Tensor<T, TGrad> operator /(Tensor<T, TGrad> left, Tensor<T, TGrad> right) => new TensorOperationTwo<T, DivOp<T, TGrad>, TGrad>(left, right);

    }
}
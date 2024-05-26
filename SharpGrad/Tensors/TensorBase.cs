using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    public abstract class TensorBase<T, TGrad>(Shape shape) : ITensor<TensorBase<T, TGrad>, T>
        where T : unmanaged, INumber<T>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
    {
        public static readonly TensorBase<T, TGrad> Empty = new Tensor<T, TGrad>();
        internal abstract AcceleratorBuffer<T> Data { get; }

        protected readonly Shape shape = shape;
        public Shape Shape => shape;
        public readonly long Length = shape.Size;

        public abstract T this[params int[] indices] { get; set; }


        public static void ExecGpu(
            Action<Index1D, ArrayView<T>, ArrayView<T>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            Action<Index1D, ArrayView<T>, ArrayView<T>> loadedKernel = Tensors.Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, result.View);
            Tensors.Accelerator.Synchronize();
        }

        public static void ExecGpu(
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> loadedKernel = Tensors.Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, right.View, result.View);
            Tensors.Accelerator.Synchronize();
        }


        public static void ExecGpu(
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> func,
            TensorBase<T, TGrad> left, TensorBase<T, TGrad> right, TensorBase<T, TGrad> result)
            => ExecGpu(func, left.Data.AcceleratorData, right.Data.AcceleratorData, result.Data.AcceleratorData);

        public static Tensor<T, TGrad> ExecTensorOnGpu(
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> func,
            TensorBase<T, TGrad> left, TensorBase<T, TGrad> right)
        {
            if (left.shape != right.shape)
                throw new ArgumentException($"Expected shapes {left.shape}, got {right.shape}");
            var result = new Tensor<T, TGrad>(left.shape);
            ExecGpu(func, left, right, result);
            return result;
        }

        public static void ExecGpu(
            OpCode[] operations,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            if(left.Length != right.Length || left.Length != result.Length)
                throw new ArgumentException($"Length mismatch: {nameof(left)}:{left.Length}, {nameof(right)}:{right.Length}, {nameof(result)}:{result.Length}");
            ExecGpu(operations, left, right, result);
        }

        public static void DynGpu(
            OpCode[] operations,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
        {
            Action<Index1D, ArrayView<OpCode>, ArrayView<T>, ArrayView<T>, ArrayView<T>> loadedKernel =
                Tensors.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OpCode>, ArrayView<T>, ArrayView<T>, ArrayView<T>>(KernelProcessUnit<T, TGrad>.Dynamic);
            loadedKernel(left.IntExtent, Tensors.Accelerator.Allocate1D(operations).View, left.View, right.View, result.View);
            Tensors.Accelerator.Synchronize();
        }

        public static void DynGpu(
            OpCode[] operations,
            Tensor<T, TGrad> left, Tensor<T, TGrad> right, Tensor<T, TGrad> result)
            => DynGpu(operations, left.Data.AcceleratorData, right.Data.AcceleratorData, result.Data.AcceleratorData);

        public static TensorBase<T, TGrad> operator +(TensorBase<T, TGrad> left, TensorBase<T, TGrad> right) => ExecTensorOnGpu(AddOp<T, TGrad>.ApplyGpu, left, right);
        public static TensorBase<T, TGrad> operator -(TensorBase<T, TGrad> left, TensorBase<T, TGrad> right) => ExecTensorOnGpu(SubOp<T, TGrad>.ApplyGpu, left, right);
        public static TensorBase<T, TGrad> operator *(TensorBase<T, TGrad> left, TensorBase<T, TGrad> right) => ExecTensorOnGpu(MulOp<T, TGrad>.ApplyGpu, left, right);
        public static TensorBase<T, TGrad> operator /(TensorBase<T, TGrad> left, TensorBase<T, TGrad> right) => ExecTensorOnGpu(DivOp<T, TGrad>.ApplyGpu, left, right);

    }
}
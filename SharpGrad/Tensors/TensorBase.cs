using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public abstract class TensorBase<T>(Shape shape) : ITensor<TensorBase<T>, T>
        where T : unmanaged, IFloatingPoint<T>
    {
        public static readonly TensorBase<T> Empty = new Tensor<T>();
        internal abstract DeviceBuffer<T> Data { get; }

        protected readonly Shape shape = shape;
        public Shape Shape => shape;
        public readonly long Length = shape.Size;

        public abstract T this[params int[] indices] { get; set; }

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
            TensorBase<T> left, TensorBase<T> right, TensorBase<T> result)
            => ExecGpu(func, left.Data.DeviceData, right.Data.DeviceData, result.Data.DeviceData);

        public static Tensor<T> ExecTensorOnGpu(
            Action<Index1D, ArrayView<T>, ArrayView<T>, ArrayView<T>> func,
            TensorBase<T> left, TensorBase<T> right)
        {
            if (left.shape != right.shape)
                throw new ArgumentException($"Expected shapes {left.shape}, got {right.shape}");
            var result = new Tensor<T>(left.shape);
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
                Tensors.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OpCode>, ArrayView<T>, ArrayView<T>, ArrayView<T>>(KernelProcessUnit<T>.Dynamic);
            loadedKernel(left.IntExtent, Tensors.Accelerator.Allocate1D(operations).View, left.View, right.View, result.View);
            Tensors.Accelerator.Synchronize();
        }

        public static void DynGpu(
            OpCode[] operations,
            Tensor<T> left, Tensor<T> right, Tensor<T> result)
            => DynGpu(operations, left.Data.DeviceData, right.Data.DeviceData, result.Data.DeviceData);

        public static TensorBase<T> operator +(TensorBase<T> left, TensorBase<T> right) => ExecTensorOnGpu(AddOp<T>.Apply, left, right);
        public static TensorBase<T> operator -(TensorBase<T> left, TensorBase<T> right) => ExecTensorOnGpu(SubOp<T>.Apply, left, right);
        public static TensorBase<T> operator *(TensorBase<T> left, TensorBase<T> right) => ExecTensorOnGpu(MulOp<T>.Apply, left, right);
        public static TensorBase<T> operator /(TensorBase<T> left, TensorBase<T> right) => ExecTensorOnGpu(DivOp<T>.Apply, left, right);

    }
}
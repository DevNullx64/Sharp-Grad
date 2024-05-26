using ILGPU;
using ILGPU.Runtime;
using System;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public abstract class TensorBase<TType>(Shape shape) where TType : unmanaged, IFloatingPoint<TType>
    {
        protected abstract DeviceBuffer<TType> Data { get; }

        protected readonly Shape shape = shape;

        public static void ExecGpu(
            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            MemoryBuffer1D<TType, Stride1D.Dense> left, MemoryBuffer1D<TType, Stride1D.Dense> right, MemoryBuffer1D<TType, Stride1D.Dense> result)
        {
            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel = Tensors.Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, right.View, result.View);
            Tensors.Accelerator.Synchronize();
        }

        public static Tensor<TType> ExecGpu(
            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            Tensor<TType> left, Tensor<TType> right)
        {
            if (left.shape != right.shape)
                throw new ArgumentException($"Expected shapes {left.shape}, got {right.shape}");
            var result = new Tensor<TType>(left.shape);
            ExecGpu(func, left, right, result);
            return result;
        }

        public static void ExecGpu(
            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            Tensor<TType> left, Tensor<TType> right, Tensor<TType> result)
            => ExecGpu(func, left.Data.DeviceData, right.Data.DeviceData, result.Data.DeviceData);

        public static void ExecGpu(
            OpCode[] operations,
            MemoryBuffer1D<TType, Stride1D.Dense> left, MemoryBuffer1D<TType, Stride1D.Dense> right, MemoryBuffer1D<TType, Stride1D.Dense> result)
        {
            if(left.Length != right.Length || left.Length != result.Length)
                throw new ArgumentException($"Length mismatch: {nameof(left)}:{left.Length}, {nameof(right)}:{right.Length}, {nameof(result)}:{result.Length}");
            ExecGpu(operations, left, right, result);
        }

        public static void DynGpu(
            OpCode[] operations,
            Tensor<TType> left, Tensor<TType> right, Tensor<TType> result)
        {
            Action<Index1D, ArrayView<OpCode>, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel =
                Tensors.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OpCode>, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>>(KernelProcessUnit<TType>.Dynamic);
            loadedKernel(left.Data.DeviceData.IntExtent, Tensors.Accelerator.Allocate1D(operations).View, left.Data.DeviceData.View, right.Data.DeviceData.View, result.Data.DeviceData.View);
            Tensors.Accelerator.Synchronize();


        }
    }
}
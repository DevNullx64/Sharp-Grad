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

        public static Tensor<TType> ExecGpu(Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
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
        {
            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel = Tensors.Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.Data.DeviceData.IntExtent, left.Data.DeviceData.View, right.Data.DeviceData.View, result.Data.DeviceData.View);
            Tensors.Accelerator.Synchronize();
        }
        public static Tensor<TType> ExecGpu(OpCode[] operations,
            Tensor<TType> left, Tensor<TType> right)
        {
            if (left.shape != right.shape)
                throw new ArgumentException($"Expected shapes {left.shape}, got {right.shape}");

            var result = new Tensor<TType>(left.shape);
            ExecGpu(operations, left, right, result);
            return result;
        }
        public static void ExecGpu(
            OpCode[] operations,
            Tensor<TType> left, Tensor<TType> right, Tensor<TType> result)
        {
            MemoryBuffer1D<OpCode, Stride1D.Dense> opsOnDevice = Tensors.Accelerator.Allocate1D(operations);
            MemoryBuffer1D<TType, Stride1D.Dense> leftOnDevice = left.Data.DeviceData;
            MemoryBuffer1D<TType, Stride1D.Dense> rightOnDevice = right.Data.DeviceData;
            MemoryBuffer1D<TType, Stride1D.Dense> resultOnDevice = result.Data.DeviceData;

            Action<Index1D, ArrayView<OpCode>, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel =
                Tensors.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OpCode>, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>>(KernelProcessUnit<TType>.Dynamic);
            loadedKernel(left.Data.DeviceData.IntExtent, opsOnDevice.View, leftOnDevice.View, rightOnDevice.View, resultOnDevice.View);
            Tensors.Accelerator.Synchronize();
        }
    }
}
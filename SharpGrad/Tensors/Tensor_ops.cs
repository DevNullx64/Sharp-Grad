using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace SharpGrad.Tensors
{
    public partial class Tensor<TType> :
        IAdditionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        ISubtractionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        IMultiplyOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        IDivisionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static void ExecGpu(
            OpCode[] operations,
            Tensor<TType> left, Tensor<TType> right, Tensor<TType> result)
        {
            MemoryBuffer1D<OpCode, Stride1D.Dense> opsOnDevice = Tensors.Accelerator.Allocate1D(operations);
            MemoryBuffer1D<TType, Stride1D.Dense> leftOnDevice = left.data_.DeviceData;
            MemoryBuffer1D<TType, Stride1D.Dense> rightOnDevice = right.data_.DeviceData;
            MemoryBuffer1D<TType, Stride1D.Dense> resultOnDevice = result.data_.DeviceData;

            Action<Index1D, ArrayView<OpCode>, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel =
                Tensors.Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OpCode>, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>>(KernelProcessUnit<TType>.Dynamic);
            loadedKernel(left.data_.DeviceData.IntExtent, opsOnDevice.View, leftOnDevice.View, rightOnDevice.View, resultOnDevice.View);
            Tensors.Accelerator.Synchronize();

            resultOnDevice.CopyToCPU(result.data_);
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
            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            Tensor<TType> left, Tensor<TType> right, Tensor<TType> result)
        {
            MemoryBuffer1D<TType, Stride1D.Dense> leftOnDevice = left.data_.DeviceData;
            MemoryBuffer1D<TType, Stride1D.Dense> rightOnDevice = right.data_.DeviceData;
            MemoryBuffer1D<TType, Stride1D.Dense> resultOnDevice = result.data_.DeviceData;

            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel = Tensors.Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.data_.DeviceData.IntExtent, leftOnDevice.View, rightOnDevice.View, resultOnDevice.View);
            Tensors.Accelerator.Synchronize();

            var r = result.data_.CPUData;
        }

        public static Tensor<TType> ExecGpu(Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            Tensor<TType> left, Tensor<TType> right)
        {
            if (left.shape != right.shape)
                throw new ArgumentException($"Expected shapes {left.shape}, got {right.shape}");

            var result = new Tensor<TType>(left.shape);
            ExecGpu(func, left, right, result);
            return result;
        }

        public static Tensor<TType> operator +(Tensor<TType> left, Tensor<TType> right) => ExecGpu(AddOp<TType>.Apply, left, right);

        public static Tensor<TType> operator -(Tensor<TType> left, Tensor<TType> right) => ExecGpu(SubOp<TType>.Apply, left, right);

        public static Tensor<TType> operator *(Tensor<TType> left, Tensor<TType> right) => ExecGpu(MulOp<TType>.Apply, left, right);

        public static Tensor<TType> operator /(Tensor<TType> left, Tensor<TType> right) => ExecGpu(DivOp<TType>.Apply, left, right);
    }
}
using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.OpenCL;
using System.Diagnostics;
using System.Xml.Serialization;
using ILGPU.Runtime.Cuda;

namespace SharpGrad.Tensors
{
    public static class Kernel<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static void Add(Index1D i, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[i] = left[i] + right[i];
        public static void Sub(Index1D i, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[i] = left[i] - right[i];
        public static void Mul(Index1D i, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[i] = left[i] * right[i];
        public static void Div(Index1D i, ArrayView<TType> left, ArrayView<TType> right, ArrayView<TType> output)
            => output[i] = left[i] / right[i];
    }

    public readonly partial struct Tensor<TType> :
        IAdditionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        ISubtractionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        IMultiplyOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>,
        IDivisionOperators<Tensor<TType>, Tensor<TType>, Tensor<TType>>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public static void ExecGpu(Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            Tensor<TType> left, Tensor<TType> right, Tensor<TType> result)
        {
            using Context context = Context.Create(builder => builder.AllAccelerators());
            Debug.WriteLine("Context: " + context.ToString());

            Device device = context.GetPreferredDevice(preferCPU: false);
            using Accelerator acc = device.CreateAccelerator(context);

            acc.PrintInformation();

            MemoryBuffer1D<TType, Stride1D.Dense> leftOnDevice = acc.Allocate1D(left.data);
            MemoryBuffer1D<TType, Stride1D.Dense> rightOnDevice = acc.Allocate1D(right.data);
            MemoryBuffer1D<TType, Stride1D.Dense> resultOnDevice = acc.Allocate1D(result.data);

            Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> loadedKernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>>(func);
            loadedKernel(left.data.Length, leftOnDevice.View, rightOnDevice.View, resultOnDevice.View);
            acc.Synchronize();

            resultOnDevice.CopyToCPU(result.data);
        }

        public static Tensor<TType> ExecGpu(Action<Index1D, ArrayView<TType>, ArrayView<TType>, ArrayView<TType>> func,
            Tensor<TType> left, Tensor<TType> right)
        {
            if (left.Shape != right.Shape)
                throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");

            var result = new Tensor<TType>(left.Shape);
            ExecGpu(func, left, right, result);
            return result;
        }

        public static Tensor<TType> operator +(Tensor<TType> left, Tensor<TType> right) => ExecGpu(Kernel<TType>.Add, left, right);

        public static Tensor<TType> operator -(Tensor<TType> left, Tensor<TType> right) => ExecGpu(Kernel<TType>.Sub, left, right);

        public static Tensor<TType> operator *(Tensor<TType> left, Tensor<TType> right) => ExecGpu(Kernel<TType>.Mul, left, right);

        public static Tensor<TType> operator /(Tensor<TType> left, Tensor<TType> right) => ExecGpu(Kernel<TType>.Div, left, right);
    }
}

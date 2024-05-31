using ILGPU;
using ILGPU.Runtime;
using System;
using System.Diagnostics;
using System.Numerics;
using System.Threading;

namespace SharpGrad.Tensors
{
    public static class Acc
    {
        private static Context GetContext()
        {
            Context result = Context.Create(builder => builder.AllAccelerators());
            Debug.WriteLine($"Context created: {result}");
            return result;
        }
        private static readonly Context context = GetContext();

        private static Device GetDevice(Context context)
        {
            Device result = context.GetPreferredDevice(preferCPU: false);
            Debug.WriteLine($"Device created: {result}");
            return result;
        }
        private static readonly Device device = GetDevice(context);
        public static readonly Accelerator Accelerator = device.CreateAccelerator(context);

        private static void FillKernel<T>(Index1D idx, ArrayView1D<T, Stride1D.Dense> view, T value)
            where T : unmanaged, INumber<T> { view[idx] = value; }
        public static void Fill<TFrom, TTo, Grad>(this MemoryBuffer1D<TTo, Stride1D.Dense> mem, TTo value)
            where TFrom : unmanaged, INumber<TFrom>
            where TTo : unmanaged, INumber<TTo>
            where Grad : unmanaged, IFloatingPoint<Grad>
        {
            Action<Index1D, ArrayView1D<TTo, Stride1D.Dense>, TTo> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<TTo, Stride1D.Dense>, TTo>(FillKernel);
            loadedKernel(mem.IntExtent, mem.View, value);
            Accelerator.Synchronize();
        }

        public static void Fill<TFrom, TTo, Grad>(this Tensor<TFrom, TTo, Grad> tensor, TTo value)
            where TFrom : unmanaged, INumber<TFrom>
            where TTo : unmanaged, INumber<TTo>
            where Grad : unmanaged, IFloatingPoint<Grad>
        {
            Fill<TFrom, TTo, Grad>(tensor.Data, value);
        }

        public static void Exec<TFrom, TTo>(
            Action<Index1D, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TTo, Stride1D.Dense>> func,
            MemoryBuffer1D<TFrom, Stride1D.Dense> left, MemoryBuffer1D<TTo, Stride1D.Dense> result)
            where TFrom : unmanaged, INumber<TFrom>
            where TTo : unmanaged, INumber<TTo>
        {
            Action<Index1D, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TTo, Stride1D.Dense>> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, result.View);
            Accelerator.Synchronize();
        }

        public static void Exec<TFrom, TTo>(
            Action<Index1D, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TTo, Stride1D.Dense>> func,
            MemoryBuffer1D<TFrom, Stride1D.Dense> left, MemoryBuffer1D<TFrom, Stride1D.Dense> right, MemoryBuffer1D<TTo, Stride1D.Dense> result)
            where TFrom : unmanaged, INumber<TFrom>
            where TTo : unmanaged, INumber<TTo>
        {
            Action<Index1D, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TTo, Stride1D.Dense>> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, right.View, result.View);
            Accelerator.Synchronize();
        }


        public static void Exec<TFrom, TTo, TGrad>(
            Action<Index1D, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TTo, Stride1D.Dense>> func,
            Tensor<TFrom, TFrom, TGrad> left, Tensor<TFrom, TFrom, TGrad> right, Tensor<TFrom, TTo, TGrad> result)
            where TFrom : unmanaged, INumber<TFrom>
        where TTo : unmanaged, INumber<TTo>
        where TGrad : unmanaged, IFloatingPoint<TGrad>
        {
            if (left.Shape != right.Shape)
                throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");

            Exec(func, left.Data.AcceleratorData, right.Data.AcceleratorData, result.Data.AcceleratorData);
        }

        public static DataTensor<TFrom, TTo, TGrad> Exec<TFrom, TTo, TGrad>(
            Action<Index1D, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TFrom, Stride1D.Dense>, ArrayView1D<TTo, Stride1D.Dense>> func,
            Tensor<TFrom, TFrom, TGrad> left, Tensor<TFrom, TFrom, TGrad> right)
            where TFrom : unmanaged, INumber<TFrom>
            where TTo : unmanaged, INumber<TTo>
            where TGrad : unmanaged, IFloatingPoint<TGrad>
        {
            if (left.Shape != right.Shape)
                throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");
            var result = new DataTensor<TFrom, TTo, TGrad>(left.Shape);
            Exec(func, left, right, result);
            return result;
        }

        public static void Exec<T>(
            OpCode[] operations,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>
        {
            if (left.Length != right.Length || left.Length != result.Length)
                throw new ArgumentException($"Length mismatch: {nameof(left)}:{left.Length}, {nameof(right)}:{right.Length}, {nameof(result)}:{result.Length}");
            Exec(operations, left, right, result);
        }

    }
}

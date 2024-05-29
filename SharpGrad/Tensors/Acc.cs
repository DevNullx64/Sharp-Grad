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

        private static void SetKernel<T>(Index1D idx, ArrayView1D<T, Stride1D.Dense> view, T value)
            where T : unmanaged, INumber<T> { view[idx] = value; }
        public static void Fill<T, Grad>(this MemoryBuffer1D<T, Stride1D.Dense> mem, T value)
            where T : unmanaged, INumber<T>
            where Grad : unmanaged, IFloatingPoint<Grad>
        {
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, T> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<T, Stride1D.Dense>, T>(SetKernel);
            loadedKernel(mem.IntExtent, mem.View, value);
            Accelerator.Synchronize();
        }

        public static void Fill<T, Grad>(this Tensor<T, Grad> tensor, T value)
            where T : unmanaged, INumber<T>
            where Grad : unmanaged, IFloatingPoint<Grad>
        {
            Fill<T, Grad>(tensor.Data, value);
        }

        public static void Exec<T>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>
        {
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, result.View);
            Accelerator.Synchronize();
        }

        public static void Exec<T>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> func,
            MemoryBuffer1D<T, Stride1D.Dense> left, MemoryBuffer1D<T, Stride1D.Dense> right, MemoryBuffer1D<T, Stride1D.Dense> result)
            where T : unmanaged, INumber<T>
        {
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> loadedKernel = Accelerator.LoadAutoGroupedStreamKernel(func);
            loadedKernel(left.IntExtent, left.View, right.View, result.View);
            Accelerator.Synchronize();
        }


        public static void Exec<T, TGrad>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> func,
        Tensor<T, TGrad> left, Tensor<T, TGrad> right, Tensor<T, TGrad> result)
            where T : unmanaged, INumber<T>
            where TGrad : unmanaged, IFloatingPoint<TGrad>
        {
            if(left.Shape != right.Shape)
                throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");

            Exec(func, left.Data.AcceleratorData, right.Data.AcceleratorData, result.Data.AcceleratorData);
        }

        public static DataTensor<T, TGrad> Exec<T, TGrad>(
            Action<Index1D, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>, ArrayView1D<T, Stride1D.Dense>> func,
            Tensor<T, TGrad> left, Tensor<T, TGrad> right)
            where T : unmanaged, INumber<T>
            where TGrad : unmanaged, IFloatingPoint<TGrad>
        {
            if (left.Shape != right.Shape)
                throw new ArgumentException($"Expected shapes {left.Shape}, got {right.Shape}");
            var result = new DataTensor<T, TGrad>(left.Shape);
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

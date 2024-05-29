using ILGPU;
using ILGPU.Runtime;
using System;
using System.Diagnostics;
using System.Numerics;

namespace SharpGrad.Tensors
{
    public static class Tensors
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
    }
}

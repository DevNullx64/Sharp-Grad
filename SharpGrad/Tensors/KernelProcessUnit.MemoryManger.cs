using ILGPU.Runtime;
using ILGPU;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit
    {
        private static List<AcceleratorBuffer> Allocs = [];
        internal static MemoryBuffer1D<T, Stride1D.Dense> MemoryBuffer1D<T>(long length)
            where T : unmanaged, INumber<T>
        {
            try
            {
                return Accelerator.Allocate1D<T, Stride1D.Dense>(length, new Stride1D.Dense());
            }
            catch { }
            FreeAcceleratorMemory(length);
            return MemoryBuffer1D<T>(length);
        }
        public static void FreeAcceleratorMemory(long length = 0)
        {
            lock (Allocs)
            {
                if (Allocs.Count == 0)
                    return;
                long toFree = length < 1 ? long.MaxValue : length;
                foreach (var buf in Allocs.Where(e => e.Location == BufferLocation.Accelerator).OrderBy(e => e.LastAccess))
                {
                    buf.Location = BufferLocation.Ram;
                    toFree -= buf.Length;
                    if (toFree <= 0)
                        break;
                }
                Synchronize();
            }
        }

    }
}

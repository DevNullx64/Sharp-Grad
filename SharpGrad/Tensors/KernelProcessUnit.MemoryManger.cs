using ILGPU.Runtime;
using ILGPU;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace SharpGrad.Tensors
{
    public partial class KernelProcessUnit
    {
        private List<AcceleratorBuffer> Allocs = [];
        internal MemoryBuffer1D<T, Stride1D.Dense> MemoryBuffer1D<T>(long length)
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

        internal MemoryBuffer1D<T, Stride1D.Dense> MemoryBuffer1D<T>(T[] values)
            where T : unmanaged, INumber<T>
        {
            var result = MemoryBuffer1D<T>(values.LongLength);
            result.CopyFromCPU(values);
            return result;
        }

        public long FreeAcceleratorMemory(long length = 0)
        {
            lock (Allocs)
            {
                if (Allocs.Count == 0)
                    return 0;

                long toFree = length < 1 ? long.MaxValue : length;
                long Freed = 0;
                foreach (var buf in Allocs.Where(e => e.Location == BufferLocation.Accelerator).OrderBy(e => e.LastAccess))
                {
                    buf.Location = BufferLocation.Ram;
                    Freed += buf.Length;
                    if (Freed >= toFree)
                        break;
                }
                Synchronize();
                
                return Freed;
            }
        }

        private readonly HashSet<MemoryBuffer> MemoryBuffers = [];
        internal void Dispose(AcceleratorBuffer acceleratorBuffer)
            => Allocs.Remove(acceleratorBuffer);

        internal void Dispose<T>(AcceleratorBuffer<T> acceleratorBuffer)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
            => Allocs.Remove(acceleratorBuffer);

        private MemoryBuffer1D<T, Stride1D.Dense> Allocate1D<T>(long length)
            where T : unmanaged
        {
            try
            {
                Stride1D.Dense stride = default;
                MemoryBuffer1D<T, Stride1D.Dense> buffer = Accelerator.Allocate1D<T, Stride1D.Dense>(length, stride);
                MemoryBuffers.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
            }

            FreeAcceleratorMemory(length);
            return Allocate1D<T>(length);
        }
        private MemoryBuffer1D<T, Stride1D.Dense> Allocate1D<T>(T[] data)
            where T : unmanaged, INumber<T>
        {
            try
            {
                MemoryBuffer1D<T, Stride1D.Dense> buffer = Allocate1D<T>(data.LongLength);
                buffer.CopyFromCPU(data);
                MemoryBuffers.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }

        public AcceleratorBuffer<T> GetAcceleratorBuffer<T>(long length)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = AcceleratorBuffer<T>.Create(length);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }
        public AcceleratorBuffer<T> GetAcceleratorBuffer<T>(T[] values)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = AcceleratorBuffer<T>.Create(values);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }


        public AcceleratorBuffer<T> GetAcceleratorBuffer<T>(MemoryBuffer1D<T, Stride1D.Dense> data)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = AcceleratorBuffer<T>.Create(data);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }


    }
}

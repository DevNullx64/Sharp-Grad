﻿using ILGPU.Runtime;
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
    public interface ILowLevelMemoryManager
    {
        internal MemoryBuffer1D<T, Stride1D.Dense> MemoryBuffer1D<T>(long length)
            where T : unmanaged, INumber<T>;

        internal MemoryBuffer1D<T, Stride1D.Dense> MemoryBuffer1D<T>(T[] values)
            where T : unmanaged, INumber<T>;
        AcceleratorBuffer<T> GetBuffer<T>(MemoryBuffer1D<T, Stride1D.Dense> data)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>;

        void Fill<T>(MemoryBuffer1D<T, Stride1D.Dense> acceleratorData, T value)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>;
    }

    public interface IBufferManager
    {
        AcceleratorBuffer<T> GetBuffer<T>(long length)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>;

        AcceleratorBuffer<T> GetBuffer<T>(T[] values)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>;

        void Release(AcceleratorBuffer buffer);

        long OffloadMemory(long length = 0);

        void Synchronize();
    }
    public interface IMemoryManager: IBufferManager, ILowLevelMemoryManager
    { }


    public partial class KernelProcessUnit : IMemoryManager
    {
        private readonly List<AcceleratorBuffer> Allocs = [];
        MemoryBuffer1D<T, Stride1D.Dense> ILowLevelMemoryManager.MemoryBuffer1D<T>(long length)
        {
            int retry = 3;
            while (retry-- > 0)
            {
                try { return Accelerator.Allocate1D<T, Stride1D.Dense>(length, new Stride1D.Dense()); }
                catch { }
                ((IMemoryManager)this).OffloadMemory(length);
            }
            throw new OutOfMemoryException($"Failed to allocate memory for {length} elements.");
        }

        MemoryBuffer1D<T, Stride1D.Dense> ILowLevelMemoryManager.MemoryBuffer1D<T>(T[] values)
        {
            var result = ((IMemoryManager)this).MemoryBuffer1D<T>(values.LongLength);
            result.CopyFromCPU(values);
            return result;
        }

        long IBufferManager.OffloadMemory(long length)
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

        public AcceleratorBuffer<T> GetBuffer<T>(long length)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = new(this, length);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }
        public AcceleratorBuffer<T> GetBuffer<T>(T[] values)
            where T : unmanaged, INumber<T>, IPowerFunctions<T>
        {
            try
            {
                AcceleratorBuffer<T> buffer = new(this, values);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }


        AcceleratorBuffer<T> ILowLevelMemoryManager.GetBuffer<T>(MemoryBuffer1D<T, Stride1D.Dense> data)        {
            try
            {
                AcceleratorBuffer<T> buffer = new(this, data);
                Allocs.Add(buffer);
                return buffer;
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                throw new Exception("Failed to create buffer from data.", e);
            }
        }

        public void Release(AcceleratorBuffer buffer)
            => Allocs.Remove(buffer);

        private static void FillKernel<T>(LongIndex1D index1D, ArrayView<T> buffer, T value)
            where T : unmanaged, INumber<T>
        { buffer[index1D] = value; }

        void ILowLevelMemoryManager.Fill<T>(MemoryBuffer1D<T, Stride1D.Dense> acceleratorData, T value)
        {
            var kernel = Accelerator.LoadAutoGroupedStreamKernel<LongIndex1D, ArrayView<T>, T>(FillKernel);
            kernel(acceleratorData.Length, acceleratorData.View, value);
            throw new NotImplementedException();
        }

    }
}
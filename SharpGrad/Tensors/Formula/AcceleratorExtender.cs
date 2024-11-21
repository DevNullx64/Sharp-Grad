using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Memory;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace SharpGrad.Tensors.Formula
{
    public class AcceleratorExtender : IBufferManager
    {
        private static Device SelectDefaultDevice()
        {
            Device? result = null;
            foreach (var device in Context.Devices)
            {
                if (device.AcceleratorType != AcceleratorType.CPU)
                {
                    if (result is null)
                        result = device;
                    else
                    {
                        if (result.AcceleratorType == AcceleratorType.Cuda)
                        {
                            if (device.AcceleratorType == AcceleratorType.Cuda && device.MemorySize > result.MemorySize)
                                result = device;
                        }
                        else
                        {
                            if (device.AcceleratorType == AcceleratorType.Cuda)
                                result = device;
                        }
                    }
                }
            }
            return result ?? Context.GetPreferredDevice(preferCPU: true);
        }

        public static readonly Context Context = Context.Create(builder => builder.AllAccelerators());
        public static readonly Device DefaultDevice = SelectDefaultDevice();
        public static readonly Accelerator DefaultAccelerator = DefaultDevice.CreateAccelerator(Context);
        public static readonly AcceleratorExtender DefaultExtender = new(DefaultAccelerator);

        private static readonly Dictionary<Accelerator, List<AcceleratorBuffer>> MMUs = [];

        //private readonly MemoryManagementUnit Buffers;
        private readonly List<AcceleratorBuffer> Buffers;
        private IReadOnlyList<AcceleratorBuffer> BuffersSnapshot
        {
            get
            {
                BuffersLock.EnterReadLock();
                try
                {
                    return [.. Buffers];
                }
                finally
                {
                    BuffersLock.ExitReadLock();
                }
            }
        }
        private readonly ReaderWriterLockSlim BuffersLock = new();

        public readonly Accelerator Accelerator;
        public Device Device => Accelerator.Device;

        private long lastUsedAcceleratorMemory = 0;
        public long UsedAcceleratorMemory
        {
            get
            {
                if (BuffersLock.TryEnterReadLock(0))
                    try
                    {
                        long result = 0;
                        foreach (var buffer in Buffers)
                            if (buffer.SafeLocation == BufferLocation.Accelerator)
                                result += buffer.LongLength;

                        lastUsedAcceleratorMemory = result;
                        return result;
                    }
                    finally
                    {
                        BuffersLock.ExitReadLock();
                    }
                else
                    return lastUsedAcceleratorMemory;
            }
        }

        public long UsedRamMemory
        {
            get
            {
                if (BuffersLock.TryEnterReadLock(0))
                    try
                    {
                        long result = 0;
                        foreach (var buffer in Buffers)
                            if (buffer.SafeLocation == BufferLocation.Ram)
                                result += buffer.LongLength;

                        return result;
                    }
                    finally
                    {
                        BuffersLock.ExitReadLock();
                    }
                else
                    return 0;
            }
        }

        public AcceleratorExtender(Accelerator? accelerator)
        {
            Accelerator = accelerator ?? DefaultAccelerator;
            List<AcceleratorBuffer>? value;
            lock (MMUs)
            {
                if (!MMUs.TryGetValue(Accelerator, out value))
                    MMUs.Add(Accelerator, value = []);
            }
            Buffers = value;
        }

        public AcceleratorBuffer<T> Allocate<T>(long length)
            where T : unmanaged
        {
            var buffer = new AcceleratorBuffer<T>(this, length);
            BuffersLock.EnterWriteLock();
            try { Buffers.Add(buffer); }
            finally { BuffersLock.ExitWriteLock(); }
            return buffer;
        }

        public AcceleratorBuffer<T> Allocate<T>(T[] values)
            where T : unmanaged
        {
            var buffer = new AcceleratorBuffer<T>(this, values);
            BuffersLock.EnterWriteLock();
            try { Buffers.Add(buffer); }
            finally { BuffersLock.ExitWriteLock(); }
            return buffer;
        }

        internal void Release(AcceleratorBuffer acceleratorBuffer)
        {
            acceleratorBuffer.SafeLocation = BufferLocation.Empty;
            BuffersLock.EnterWriteLock();
            try { Buffers.Remove(acceleratorBuffer); }
            finally { BuffersLock.ExitWriteLock(); }
        }
        void IBufferManager.Release(AcceleratorBuffer buffer) => Release(buffer);


        public long OffloadMemory(long length = long.MaxValue, Device? device = null)
        {
            device ??= Device;
            long freed = 0;
            var bufferSnapshot = BuffersSnapshot;
            var bufferByLastAccess = bufferSnapshot
                .Where(e => e.IsNoLockHeld && e.SafeLocation == BufferLocation.Accelerator && e.SafeAccelerator.Device == device)
                .OrderBy(e => e.LastAccess);
            foreach (var buffer in bufferByLastAccess)
            {
                buffer.SafeLocation = BufferLocation.Ram;
                if (buffer.SafeLocation == BufferLocation.Ram)
                {
                    freed += buffer.LongLength;
                    if (freed >= length)
                        break;
                }
            }
            return freed;
        }

        public void Synchronize()
            => Accelerator.Synchronize();

        internal MemoryBuffer1D<T, Stride1D.Dense> Allocate1D<T>(T[] data, Accelerator? accelerator = null)
            where T : unmanaged
        {
            accelerator ??= Accelerator;
            int retry = 1;
            MemoryBuffer1D<T, Stride1D.Dense>? newBuffer = null;
            while (retry > 0)
                try
                {
                    newBuffer = accelerator.Allocate1D(data);
                    retry = 0;
                }
                catch (OutOfMemoryException)
                {
                    if (retry-- <= 0)
                        throw;
                    else
                        OffloadMemory(data.Length);
                }

            return newBuffer ?? throw new OutOfMemoryException($"Failed to allocate Buffers for {data.Length} dataElements.");
        }
        internal MemoryBuffer1D<T, Stride1D.Dense> Allocate1D<T>(long length)
            where T : unmanaged
        {
            int retry = 1;
            MemoryBuffer1D<T, Stride1D.Dense>? newBuffer = null;
            while (retry > 0)
                try
                {
                    newBuffer = Accelerator.Allocate1D<T>(length);
                    retry = 0;
                }
                catch (OutOfMemoryException)
                {
                    if (retry-- <= 0)
                        throw;
                    else
                        OffloadMemory(length);
                }

            return newBuffer;
        }
        // Kernel to fill a buffer with a @SafeAccelerator
        private static void FillKernel<T>(Index1D index1D, ArrayView<T> buffer, T value)
        where T : unmanaged
        { buffer[index1D] = value; }

        /// <inheritdoc/>
        internal void Fill<T>(MemoryBuffer1D<T, Stride1D.Dense> acceleratorData, T value, Accelerator? accelerator = null) where T : unmanaged
        {
            accelerator ??= Accelerator;
            Action<Index1D, ArrayView<T>, T> kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<T>, T>(FillKernel);
            kernel((int)acceleratorData.Length, acceleratorData.View, value);
        }
    }
}

using ILGPU;
using ILGPU.IR;
using ILGPU.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SharpGrad.Tensors
{

    /// <summary>
    /// A structure that manages data on the RAM and a <see cref="Accelerator"/> (GPU). It free the RAM data when the data is available on the <see cref="Accelerator"/>. And vice versa.
    /// </summary>
    /// <typeparam name="T">The type of the data</typeparam>
    /// <param name="length">The length of the data</param>
    /// <remarks>If only <paramref name="length"/> is provided, no memory will be allocated on the RAM or the <see cref="Accelerator"/>. Data will be allocated and set to zero at the first access.</remarks>
    public class AcceleratorBuffer<T>(long length) : IAcceleratorBuffer<T>
        where T : unmanaged, INumber<T>
    {
        public long LastAccess { get; set; } = DateTime.UtcNow.Ticks;

        /// <summary>
        /// The length of the data.
        /// </summary>
        /// <remarks>Prefer this over other properties that return the length of the data.</remarks>
        public readonly long Length = length;

        // The data on the RAM.
        private T[]? cpuData = null;

        /// <summary>
        /// Get or set the data on the RAM.
        /// </summary>
        /// <remarks>If data is not available on the RAM, it will be copied from the <see cref="Accelerator"/>. and the <see cref="Accelerator"/> data will be disposed.</remarks>
        public T[] CPUData
        {
            get
            {
                if (cpuData is null)
                {
                    cpuData = new T[Length];
                    if(sharedData is not null)
                    {
                        sharedData.Value.CopyToCPU(cpuData);
                        sharedData = null;
                        acceleratorData?.Dispose();
                        acceleratorData = null;
                    } else if (acceleratorData is not null)
                    {
                        acceleratorData.CopyToCPU(cpuData);
                        sharedData = null;
                        acceleratorData.Dispose();
                        acceleratorData = null;
                    }
                }
                return cpuData;
            }
            set
            {
                if (value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                cpuData = value;
                sharedData = null;
                acceleratorData?.Dispose();
                acceleratorData = null;
            }
        }

        private ArrayView1D<T, Stride1D.Dense>? sharedData = null;
        public ArrayView1D<T, Stride1D.Dense> SharedData
        {
            get
            {
                if (sharedData is null)
                {
                    sharedData = SharedMemory.Allocate1D<T>((int)Length);
                    if (cpuData is not null)
                    {
                        sharedData.Value.CopyFromCPU(cpuData);
                        cpuData = null;
                        acceleratorData?.Dispose();
                        acceleratorData = null;
                    }
                    else if (acceleratorData is not null)
                    {
                        sharedData.Value.CopyFrom(acceleratorData.AsArrayView<T>(0, Length));
                        cpuData = null;
                        acceleratorData.Dispose();
                        acceleratorData = null;
                    }
                    else
                    {
                        sharedData.Value.MemSetToZero();
                    }
                }
                return sharedData.Value;
            }
            set
            {
                if(!value.IsValid)
                    throw new ArgumentException($"Invalid view");
                if(value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");

                cpuData = null;
                sharedData = value;
                acceleratorData = null;
            }
        }
        
        public ArrayView1D<T, Stride1D.Dense> CurrentView => acceleratorData is not null ? AcceleratorData.View : SharedData;

        // The data on the Accelerator.
        private MemoryBuffer1D<T, Stride1D.Dense>? acceleratorData = null;
        /// <summary>
        /// Get or set the data on the <see cref="Accelerator"/>.
        /// </summary>
        /// <remarks>If data is not available on the <see cref="Accelerator"/>, it will be copied from the RAM. and the RAM data will be disposed.</remarks>
        public MemoryBuffer1D<T, Stride1D.Dense> AcceleratorData
        {
            get
            {
                if (acceleratorData is null)
                {
                    acceleratorData = Acc.Accelerator.Allocate1D<T>(Length);
                    if (cpuData is not null)
                    {
                        acceleratorData = Acc.Accelerator.Allocate1D(CPUData);
                        cpuData = null;
                        sharedData = null;
                    }
                    else if (sharedData is not null)
                    {
                        sharedData.Value.CopyTo(acceleratorData.AsArrayView<T>(0, Length));
                        cpuData = null;
                        sharedData = null;
                    }
                    else
                        acceleratorData.MemSetToZero();
                }
                return acceleratorData;
            }
            set
            {
                if (value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                acceleratorData = value;
                cpuData = null;
            }
        }

        // Implementing and hide the IReadOnlyList<TType> interface.
        int IReadOnlyCollection<T>.Count => (int)Length;

        public bool IsEmpty => cpuData is null && sharedData is null && acceleratorData is null;
        public bool IsOnRAM => cpuData is not null;
        public bool IsOnSharedMemory => sharedData is not null;
        public bool IsOnAccelerator => acceleratorData is not null;
        public BufferLocation Location => IsOnRAM
            ? BufferLocation.Ram
            : IsOnSharedMemory
                ? BufferLocation.SharedMemory
                : IsOnAccelerator
                    ? BufferLocation.Accelerator
                    : BufferLocation.Empty;

        // Implementing and hide the IReadOnlyList<TType> interface.
        public T this[int index]
        {
            get => CPUData[index];
            set => CPUData[index] = value;
        }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
        public AcceleratorBuffer(T[] data)
            : this(data.Length) { cpuData = data; }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the <see cref="Accelerator"/>.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the RAM.</remarks>
        public AcceleratorBuffer(MemoryBuffer1D<T, Stride1D.Dense> data)
            : this(data.Length) { acceleratorData = data; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IEnumerator<T> GetEnumerator() => CPUData.AsEnumerable().GetEnumerator();
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public void Dispose()
        {
            cpuData = null;
            sharedData = null;
            acceleratorData?.Dispose();
            acceleratorData = null;
            GC.SuppressFinalize(this);
        }

        public void Fill(T value)
        {
                Array.Fill(CPUData, value);
        }
        public void MemSetToZero()
        {
            if(IsOnRAM)
                Array.Clear(CPUData, 0, CPUData.Length);
            else if (IsOnSharedMemory)
                SharedData.MemSetToZero();
            else if (IsOnAccelerator)
                AcceleratorData.MemSetToZero();
        }
        public static implicit operator T[](AcceleratorBuffer<T> gpu) => gpu.CPUData;
        public static implicit operator MemoryBuffer1D<T, Stride1D.Dense>(AcceleratorBuffer<T> gpu) => gpu.AcceleratorData;
    }
}
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
    /// Interface to manage data on the RAM and a device (GPU).
    /// </summary>
    /// <typeparam name="T"The type of the data </typeparam>
    public interface IDeviceBuffer<T> : IReadOnlyList<T>
        where T : unmanaged, IFloatingPoint<T>
    {
        /// <summary>
        /// Return the C# managed data.
        /// </summary>
        /// <remarks>If data is not available on the RAM, it will be copied from the device. Than, the shared memory and the device data will be disposed.</remarks>
        T[] CPUData { get; set; }
        /// <summary>
        /// Return the shared memory data.
        /// </summary>
        /// <remarks>If data is not available on the shared memory, it will be copied from the CPU or the device. Then, the RAM and the device data will be disposed.</remarks>
        ArrayView<T> SharedData { get; }
        /// <summary>
        /// Return the device data view from <see cref="DeviceData"/> or <see cref="SharedData"/>.
        /// </summary>
        /// <remarks>No movement of data will be done. If no data is available, SharedData will be initialized to 0 and returned.</remarks>
        ArrayView<T> View { get; }
        /// <summary>
        /// Return the device data.
        /// </summary>
        /// <remarks>If data is not available on the device, it will be copied from <see cref="CPUData"/> or <see cref="SharedData"/>. Then, the RAM and the shared memory data will be disposed.</remarks>
        MemoryBuffer1D<T, Stride1D.Dense> DeviceData { get; set; }
    }

    /// <summary>
    /// A structure that manages data on the RAM and a device (GPU). It free the RAM data when the data is available on the device. And vice versa.
    /// </summary>
    /// <typeparam name="T">The type of the data</typeparam>
    /// <param name="length">The length of the data</param>
    /// <remarks>If only <paramref name="length"/> is provided, no memory will be allocated on the RAM or the device. Data will be allocated and set to zero at the first access.</remarks>
    public class DeviceBuffer<T>(long length) : IDeviceBuffer<T>
        where T : unmanaged, IFloatingPoint<T>
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
        /// <remarks>If data is not available on the RAM, it will be copied from the device. and the device data will be disposed.</remarks>
        public T[] CPUData
        {
            get
            {
                if (cpuData is null)
                {
                    cpuData = new T[Length];
                    if(shared is not null)
                    {
                        shared.Value.CopyToCPU(cpuData);
                        shared = null;
                        deviceData?.Dispose();
                        deviceData = null;
                    } else if (deviceData is not null)
                    {
                        deviceData.CopyToCPU(cpuData);
                        shared = null;
                        deviceData.Dispose();
                        deviceData = null;
                    }
                }
                return cpuData;
            }
            set
            {
                if (value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                cpuData = value;
                shared = null;
                deviceData?.Dispose();
                deviceData = null;
            }
        }

        private ArrayView<T>? shared;
        public ArrayView<T> SharedData
        {
            get
            {
                if (shared is null)
                {
                    shared = SharedMemory.Allocate1D<T>((int)Length);
                    if (cpuData is not null)
                    {
                        shared.Value.CopyFromCPU(cpuData);
                        cpuData = null;
                        deviceData?.Dispose();
                        deviceData = null;
                    }
                    else if (deviceData is not null)
                    {
                        shared.Value.CopyFrom(deviceData.AsArrayView<T>(0, Length));
                        cpuData = null;
                        deviceData.Dispose();
                        deviceData = null;
                    }
                    else
                    {
                        shared.Value.MemSetToZero();
                    }
                }
                return shared.Value;
            }
            set
            {
                if(!value.IsValid)
                    throw new ArgumentException($"Invalid view");
                if(value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");

                cpuData = null;
                shared = value;
                deviceData = null;
            }
        }
        
        public ArrayView<T> View => deviceData is not null ? (ArrayView<T>)DeviceData.View : SharedData;

        // The data on the device.
        private MemoryBuffer1D<T, Stride1D.Dense>? deviceData = null;
        /// <summary>
        /// Get or set the data on the device.
        /// </summary>
        /// <remarks>If data is not available on the device, it will be copied from the RAM. and the RAM data will be disposed.</remarks>
        public MemoryBuffer1D<T, Stride1D.Dense> DeviceData
        {
            get
            {
                if (deviceData is null)
                {
                    deviceData = Tensors.Accelerator.Allocate1D<T>(Length);
                    if (cpuData is not null)
                    {
                        deviceData = Tensors.Accelerator.Allocate1D(CPUData);
                        cpuData = null;
                        shared = null;
                    }
                    else if (shared is not null)
                    {
                        shared.Value.CopyTo(deviceData.AsArrayView<T>(0, Length));
                        cpuData = null;
                        shared = null;
                    }
                    else
                        deviceData.MemSetToZero();
                }
                return deviceData;
            }
            set
            {
                if (value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                deviceData = value;
                cpuData = null;
            }
        }

        // Implementing and hide the IReadOnlyList<TType> interface.
        int IReadOnlyCollection<T>.Count => (int)Length;


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
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the device.</remarks>
        public DeviceBuffer(T[] data)
            : this(data.Length) { cpuData = data; }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the device.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the RAM.</remarks>
        public DeviceBuffer(MemoryBuffer1D<T, Stride1D.Dense> data)
            : this(data.Length) { deviceData = data; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IEnumerator<T> GetEnumerator() => CPUData.AsEnumerable().GetEnumerator();
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public static implicit operator T[](DeviceBuffer<T> gpu) => gpu.CPUData;
        public static implicit operator MemoryBuffer1D<T, Stride1D.Dense>(DeviceBuffer<T> gpu) => gpu.DeviceData;
    }
}
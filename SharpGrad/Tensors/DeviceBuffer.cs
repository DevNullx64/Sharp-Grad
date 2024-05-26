using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// Interface to manage data on the RAM and a device (GPU).
    /// </summary>
    /// <typeparam name="TType"The type of the data </typeparam>
    public interface IDeviceBuffer<TType> : IReadOnlyList<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        TType[] CPUData { get; set; }
        MemoryBuffer1D<TType, Stride1D.Dense> DeviceData { get; set; }
    }

    /// <summary>
    /// A structure that manages data on the RAM and a device (GPU). It free the RAM data when the data is available on the device. And vice versa.
    /// </summary>
    /// <typeparam name="TType">The type of the data</typeparam>
    /// <param name="length">The length of the data</param>
    /// <remarks>If only <paramref name="length"/> is provided, no memory will be allocated on the RAM or the device. Data will be allocated and set to zero at the first access.</remarks>
    public class DeviceBuffer<TType>(long length) : IDeviceBuffer<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        /// <summary>
        /// The length of the data.
        /// </summary>
        /// <remarks>Prefer this over other properties that return the length of the data.</remarks>
        public readonly long Length = length;

        // The data on the RAM.
        private TType[]? cpuData = null;

        /// <summary>
        /// Get or set the data on the RAM.
        /// </summary>
        /// <remarks>If data is not available on the RAM, it will be copied from the device. and the device data will be disposed.</remarks>
        public TType[] CPUData
        {
            get
            {
                if (cpuData is null)
                {
                    cpuData = new TType[Length];
                    if (deviceData is not null)
                    {
                        deviceData.CopyToCPU(cpuData);
                        deviceData.Dispose();
                        deviceData = null;
                    }
                }
                return cpuData;
            }
            set
            {
                if(value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                cpuData = value;
                deviceData?.Dispose();
                deviceData = null;
            }
        }

        // The data on the device.
        private MemoryBuffer1D<TType, Stride1D.Dense>? deviceData = null;
        /// <summary>
        /// Get or set the data on the device.
        /// </summary>
        /// <remarks>If data is not available on the device, it will be copied from the RAM. and the RAM data will be disposed.</remarks>
        public MemoryBuffer1D<TType, Stride1D.Dense> DeviceData
        {
            get
            {
                if (deviceData is null)
                {
                    deviceData = Tensors.Accelerator.Allocate1D<TType>(Length);
                    if (cpuData is not null)
                    {
                        deviceData = Tensors.Accelerator.Allocate1D<TType>(Length);
                        deviceData.CopyFromCPU(cpuData);
                        cpuData = null;
                    }
                    else
                        deviceData.MemSetToZero();
                }
                return deviceData;
            }
            set
            {
                if(value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                deviceData = value;
                cpuData = null;
            }
        }

        // Implementing and hide the IReadOnlyList<TType> interface.
        int IReadOnlyCollection<TType>.Count => (int)Length;
        // Implementing and hide the IReadOnlyList<TType> interface.
        public TType this[int index]
        {
            get => CPUData[index];
            set => CPUData[index] = value;
        }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the device.</remarks>
        public DeviceBuffer(TType[] data)
            : this(data.Length) { cpuData = data; }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the device.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the RAM.</remarks>
        public DeviceBuffer(MemoryBuffer1D<TType, Stride1D.Dense> data)
            : this(data.Length) { deviceData = data; }

        public IEnumerator<TType> GetEnumerator() => CPUData.AsEnumerable().GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public static implicit operator TType[](DeviceBuffer<TType> gpu) => gpu.CPUData;
        public static implicit operator MemoryBuffer1D<TType, Stride1D.Dense>(DeviceBuffer<TType> gpu) => gpu.DeviceData;
    }
}
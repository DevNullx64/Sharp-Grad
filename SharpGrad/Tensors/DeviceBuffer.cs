using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace SharpGrad.Tensors
{
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
    public struct DeviceBuffer<TType>(long length) : IDeviceBuffer<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        public readonly long Length = length;

        private TType[]? cpuData = null;
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

        private MemoryBuffer1D<TType, Stride1D.Dense>? deviceData = null;
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
                        deviceData = Tensors.Accelerator.Allocate1D(cpuData);
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

        [Obsolete($"Use {nameof(Length)} instead")]
        public readonly int Count => (int)Length;

        public TType this[int index] {
            get => CPUData[index];
            set => CPUData[index] = value;
        }

        public DeviceBuffer(TType[] data)
            : this(data.Length) { cpuData = data; }

        public DeviceBuffer(MemoryBuffer1D<TType, Stride1D.Dense> data)
            : this(data.Length) { deviceData = data; }

        public static implicit operator TType[](DeviceBuffer<TType> gpu) => gpu.CPUData;
        public static implicit operator MemoryBuffer1D<TType, Stride1D.Dense>(DeviceBuffer<TType> gpu) => gpu.DeviceData;

        public IEnumerator<TType> GetEnumerator() => CPUData.AsEnumerable().GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
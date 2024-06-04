using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Memory
{
    /// <summary>
    /// Interface to manage data on the RAM and a <see cref="Accelerator"/> (GPU).
    /// </summary>
    public interface IAcceleratorBuffer : IDisposable
    {
        /// <summary>
        /// Return the length of the data.
        /// </summary>
        long Length { get; }

        /// <summary>
        /// Retrun the current location of the data.
        /// </summary>
        BufferLocation Location { get; set; }

        /// <summary>
        /// Tick of the last access to the data.
        /// </summary>
        long LastAccess { get; }
    }

    /// <summary>
    /// Interface to manage data on the RAM and a <see cref="Accelerator"/> (GPU).
    /// </summary>
    /// <typeparam name="T"The type of the data </typeparam>
    public interface IAcceleratorBuffer<T> : IAcceleratorBuffer, IReadOnlyList<T>
        where T : unmanaged, INumber<T>
    {
        /// <summary>
        /// Return the C# managed data.
        /// </summary>
        /// <remarks>If data is not available on the RAM, it will be copied from the <see cref="Accelerator"/>. Than, the shared memory and the <see cref="Accelerator"/> data will be disposed.</remarks>
        T[] CPUData { get; set; }
        /// <summary>
        /// Return the <see cref="Accelerator"/> data.
        /// </summary>
        /// <remarks>If data is not available on the <see cref="Accelerator"/>, it will be copied from <see href="CPUData"/> or <see href="SharedData"/>. Then, the RAM and the shared memory data will be disposed.</remarks>
        MemoryBuffer1D<T, Stride1D.Dense> AcceleratorData { get; set; }
    }
}
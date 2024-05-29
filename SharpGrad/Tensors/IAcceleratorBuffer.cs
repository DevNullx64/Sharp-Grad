using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace SharpGrad.Tensors
{
    /// <summary>
    /// Interface to manage data on the RAM and a <see cref="Accelerator"/> (GPU).
    /// </summary>
    /// <typeparam name="T"The type of the data </typeparam>
    public interface IAcceleratorBuffer<T> : IReadOnlyList<T>, IDisposable
        where T : unmanaged, INumber<T>
    {
        /// <summary>
        /// Retrun the current location of the data.
        /// </summary>
        BufferLocation Location { get; }

        /// <summary>
        /// The buffer is resssource free.
        /// </summary>
        public bool IsEmpty { get; }

        /// <summary>
        /// The buffer is on the RAM.
        /// </summary>
        public bool IsOnRAM { get; }

        /// <summary>
        /// The buffer is on the shared memory.
        /// </summary>
        public bool IsOnSharedMemory { get; }

        /// <summary>
        /// The buffer is on the <see cref="Accelerator"/>.
        /// </summary>
        public bool IsOnAccelerator { get; }

        /// <summary>
        /// Return the C# managed data.
        /// </summary>
        /// <remarks>If data is not available on the RAM, it will be copied from the <see cref="Accelerator"/>. Than, the shared memory and the <see cref="Accelerator"/> data will be disposed.</remarks>
        T[] CPUData { get; set; }
        /// <summary>
        /// Return the shared memory data.
        /// </summary>
        /// <remarks>If data is not available on the shared memory, it will be copied from the CPU or the <see cref="Accelerator"/>. Then, the RAM and the <see cref="Accelerator"/> data will be disposed.</remarks>
        ArrayView1D<T, Stride1D.Dense> SharedData { get; }
        /// <summary>
        /// Return the <see cref="Accelerator"/> data view from <see href="AcceleratorData"> or <see href="SharedData"/>.
        /// </summary>
        /// <remarks>No movement of data will be done. If no data is available, SharedData will be initialized to 0 and returned.</remarks>
        ArrayView1D<T, Stride1D.Dense> CurrentView { get; }
        /// <summary>
        /// Return the <see cref="Accelerator"/> data.
        /// </summary>
        /// <remarks>If data is not available on the <see cref="Accelerator"/>, it will be copied from <see href="CPUData"/> or <see href="SharedData"/>. Then, the RAM and the shared memory data will be disposed.</remarks>
        MemoryBuffer1D<T, Stride1D.Dense> AcceleratorData { get; set; }
    }
}
using ILGPU;
using ILGPU.Runtime;
using SharpGrad.Tensors.Formula;
using System;
using System.Collections.Generic;

namespace SharpGrad.Memory
{
    /// <summary>
    /// Interface to manage data on the RAM and a <see cref="SafeAccelerator"/> (GPU).
    /// </summary>
    public interface IAcceleratorBuffer : IDisposable
    {
        AcceleratorExtender SafeAccelerator { get; }

        /// <summary>
        /// Tick of the last access to the data.
        /// </summary>
        long LastAccess { get; }

        /// <summary>
        /// Return the length of the data.
        /// </summary>
        long LongLength { get; }

        /// <summary>
        /// Retrun the current location of the data.
        /// </summary>
        BufferLocation SafeLocation { get; }
    }

    /// <summary>
    /// Interface to manage data on the RAM and a <see cref="Accelerator"/> (GPU).
    /// </summary>
    /// <typeparam name="T"The type of the data </typeparam>
    public interface IReadOnlyAcceleratorBuffer<T> : IAcceleratorBuffer, IReadOnlyList<T>
        where T : unmanaged
    {
        IReadOnlyList<T> SafeCPUData { get; }
    }
    public interface IUpgradableLockAcceleratorBuffer<T> : IReadOnlyAcceleratorBuffer<T>
    where T : unmanaged
    {
        IExclusiveLockAcceleratorBuffer<T> EnterExclusiveLock();
    }

    /// <summary>
    /// Interface to manage data on the RAM and a <see cref="SafeAccelerator"/> (GPU).
    /// </summary>
    /// <typeparam name="T"The type of the data </typeparam>
    public interface IExclusiveLockAcceleratorBuffer<T> : IReadOnlyAcceleratorBuffer<T>
        where T : unmanaged
    {
        new AcceleratorExtender SafeAccelerator { get; set; }

        /// <summary>
        /// Retrun the current location of the data.
        /// </summary>
        new BufferLocation SafeLocation { get; set; }

        /// <summary>
        /// Return the C# managed data.
        /// </summary>
        /// <remarks>If data is not available on the RAM, it will be copied from the <see cref="SafeAccelerator"/>. Than, the shared Buffers and the <see cref="SafeAccelerator"/> data will be disposed.</remarks>
        new T[]? SafeCPUData { get; set; }

        /// <summary>
        /// Gets the element at the specified index in the read-only list.
        /// </summary>
        /// <param name="index">The zero-based index of the element to get.</param>
        /// <returns>The element at the specified index in the read-only list.</returns>
        new T this[int index] { get; }

        ArrayView1D<T, Stride1D.Dense>? SafeAcceleratorData { get; }
    }

    internal interface IUnsafeAcceleratorBuffer : IAcceleratorBuffer
    {
        AcceleratorExtender UnsafeAccelerator { get; set; }
        BufferLocation UnsafeLocation { get; set; }

        void EnterUpgradeableLock();
        void ExitUpgradeableReadLock();
        bool IsUpgradeableReadLockHeld { get; }

        void EnterReadLock();
        void ExitReadLock();
        bool IsReadLockHeld { get; }

        void EnterExclusiveLock();
        void ExitExclusiveLock();
        bool IsExclusiveLockHeld { get; }
    }

    internal interface ILocableAcceleratorBuffer<T>: IUnsafeAcceleratorBuffer, IReadOnlyList<T>
         where T : unmanaged
    {
        T[]? UnsafeCPUData { get; set; }
        MemoryBuffer1D<T, Stride1D.Dense>? UnsafeAcceleratorData { get; }
    }
}
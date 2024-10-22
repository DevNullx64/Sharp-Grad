using ILGPU;
using ILGPU.IR;
using ILGPU.Runtime;
using SharpGrad.Tensors;
using SharpGrad.Tensors.Operators;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;

namespace SharpGrad.Memory
{
    /// <summary>
    /// Internal interface for class that can create an <see cref="AcceleratorBuffer{T}"/>.
    /// </summary>
    /// <typeparam name="T">The type of the data.</typeparam>
    internal interface ICreateAcceleratorBuffer<T>
        where T : unmanaged, INumber<T>, IPowerFunctions<T>
    {
        /// <summary>
        /// Create a new <see cref="AcceleratorBuffer{T}"/> with the specified length.
        /// </summary>
        /// <param name="length">The length of the data.</param>
        /// <returns>A new <see cref="AcceleratorBuffer{T}"/> with the specified length.</returns>
        abstract static AcceleratorBuffer<T> Create(long length);

        /// <summary>
        /// Create a new <see cref="AcceleratorBuffer{T}"/> with the specified data.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <returns>A new <see cref="AcceleratorBuffer{T}"/> with the specified data.</returns>
        abstract static AcceleratorBuffer<T> Create(T[] data);

        /// <summary>
        /// Create a new <see cref="AcceleratorBuffer{T}"/> with the specified data.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <returns>A new <see cref="AcceleratorBuffer{T}"/> with the specified data.</returns>
        abstract static AcceleratorBuffer<T> Create(AcceleratorBuffer<T> data);

        /// <summary>
        /// Create a new <see cref="AcceleratorBuffer{T}"/> with the specified data.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <returns>A new <see cref="AcceleratorBuffer{T}"/> with the specified data.</returns>
        abstract static AcceleratorBuffer<T> Create(MemoryBuffer1D<T, Stride1D.Dense> data);
    }

    /// <summary>
    /// A structure that manages data on the RAM and a <see cref="Accelerator"/> (GPU). It free the RAM data when the data is available on the <see cref="Accelerator"/>. And vice versa.
    /// </summary>
    /// <param name="length">The length of the data.</param>
    public abstract class AcceleratorBuffer(MemoryManagementUnit mmu, long length) : IAcceleratorBuffer
    {
        protected MemoryManagementUnit MemoryManager = mmu ?? throw new ArgumentNullException(nameof(mmu));
        /// <summary>
        /// The last time the data was accessed on the <see cref="Accelerator"/>.
        /// </summary>
        public long LastAccess { get; protected set; } = DateTime.UtcNow.Ticks;

        /// <summary>
        /// The length of the data.
        /// </summary>
        /// <remarks>Prefer this over other properties that return the length of the data.</remarks>
        public long Length { get; } = length;

        private bool disposedValue;

        /// <summary>
        /// The location of the data.
        /// </summary>
        public abstract BufferLocation Location { get; set; }

        /// <summary>
        /// Is the buffer empty (ressource free).
        /// </summary>
        public bool IsEmpty { get => Location == BufferLocation.Empty; }

        /// <summary>
        /// Dispose the buffer.
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    mmu.Release(this);
                }

                // TODO: libérer les ressources non managées (objets non managés) et substituer le finaliseur
                // TODO: affecter aux grands champs une valeur null
                disposedValue = true;
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }

    /// <summary>
    /// A structure that manages data on the RAM and a <see cref="Accelerator"/> (GPU). It free the RAM data when the data is available on the <see cref="Accelerator"/>. And vice versa.
    /// </summary>
    /// <typeparam name="T">The type of the data</typeparam>
    /// <remarks>If only <paramref name="length"/> is provided, no memory will be allocated on the RAM or the <see cref="Accelerator"/>. Data will be allocated and set to zero at the first access.</remarks>
    public class AcceleratorBuffer<T> : AcceleratorBuffer, IAcceleratorBuffer<T>, IReadOnlyList<T>
        where T : unmanaged
    {
        /// <summary>
        /// Get or set the threshold to force the data to be copied to the <see cref="Accelerator"/>.
        /// </summary>
        public static int CopyToAcceleratorThreshold = 512;

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
                Location = BufferLocation.Ram;
                return cpuData;
            }
            set
            {
                if (value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                cpuData = value;
                acceleratorData?.Dispose();
                acceleratorData = null;
            }
        }

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
                Location = BufferLocation.Accelerator;
                LastAccess = DateTime.UtcNow.Ticks;
                return acceleratorData;
            }
            set
            {
                if (value.Length != Length)
                    throw new ArgumentException($"Expected length {Length}, got {value.Length}");
                acceleratorData?.Dispose();
                acceleratorData = value;
                cpuData = null;
            }
        }

        // Implementing and hide the IReadOnlyList<TType> interface.
        int IReadOnlyCollection<T>.Count => (int)Length;

        // Implementing and hide the IReadOnlyList<TType> interface.
        private bool IsOnRAM => cpuData is not null;

        // Implementing and hide the IReadOnlyList<TType> interface.
        private bool IsOnAccelerator => acceleratorData is not null;

        /// <summary>
        /// Get or set the location of the data.
        /// </summary>
        /// <remarks>Set <see cref="BufferLocation.Empty"/> to free the ressources. If and empty buffer is accessed using the <see cref="AcceleratorData"/>, the data will be allocated but not set to zero.</remarks>
        public override BufferLocation Location {
            get {
                return IsOnRAM
                    ? BufferLocation.Ram
                    : IsOnAccelerator
                        ? BufferLocation.Accelerator
                        : BufferLocation.Empty;
            }
            set
            {
                if (value != Location)
                {
                    switch (value)
                    {
                        case BufferLocation.Ram:
                            cpuData ??= new T[Length];
                            if (acceleratorData is not null)
                            {
                                acceleratorData.CopyToCPU(cpuData);
                                MemoryManager.Synchronize();
                                acceleratorData.Dispose();
                                acceleratorData = null;
                            }
                            break;
                        case BufferLocation.Accelerator:
                            acceleratorData ??= ((ILowLevelMemoryManager) MemoryManager).MemoryBuffer1D<T>(Length);
                            if (cpuData is not null)
                            {
                                acceleratorData = ((ILowLevelMemoryManager)MemoryManager).MemoryBuffer1D(CPUData);
                                MemoryManager.Synchronize();
                                cpuData = null;
                            }
                            break;
                        case BufferLocation.Empty:
                            cpuData = null;
                            acceleratorData?.Dispose();
                            acceleratorData = null;
                            break;
                    }
                }
            }
        }

        /// <summary>
        /// Get the data at the specified index.
        /// </summary>
        /// <param name="index">The index of the data.</param>
        /// <returns>The data at the specified index.</returns>
        /// <remarks>Accessing data will set the <see cref="Location"/> to <see cref="BufferLocation.Ram"/>.</remarks>
        public T this[int index]
        {
            get => CPUData[index];
            set => CPUData[index] = value;
        }

        /// <summary>
        /// Get the data at the specified index.
        /// </summary>
        /// <param name="index">The index of the data.</param>
        /// <returns>The data at the specified index.</returns>
        /// <remarks>Accessing data will set the <see cref="Location"/> to <see cref="BufferLocation.Accelerator"/>.</remarks>
        public T this[Index1D index]
        {
            get => AcceleratorData.View[index];
            set => AcceleratorData.View[index] = value;
        }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
        public AcceleratorBuffer(MemoryManagementUnit mmu, long length)
            : base(mmu, length) { }

        /// <summary>
        /// Create a new DeviceBuffer with the specified length.
        /// </summary>
        /// <param name="data">The data to be copied to the RAM.</param>
        /// <remarks><paramref name="data"/> will be copied as reference. But this link will be broken when the data is copied to the <see cref="Accelerator"/>.</remarks>
        public AcceleratorBuffer(MemoryManagementUnit mmu, T[] data)
            : this(mmu, data.Length)
        {
            ArgumentNullException.ThrowIfNull(data);
            cpuData = data;
        }
        public AcceleratorBuffer(MemoryManagementUnit mmu, MemoryBuffer1D<T, Stride1D.Dense> data)
            : this(mmu, data.Length) {
            if (data.IsDisposed)
                throw new ArgumentException($"The data is disposed.");
            acceleratorData = data; 
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IEnumerator<T> GetEnumerator() => CPUData.AsEnumerable().GetEnumerator();
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                cpuData = null;
                acceleratorData?.Dispose();
                acceleratorData = null;
                MemoryManager.Release(this);
            }
        }

        /// <summary>
        /// Check if the data is equal to the specified data.
        /// </summary>
        /// <param name="other">The data to compare with.</param>
        /// <returns>True if the data is equal to the specified data, false otherwise.</returns>
        public bool IsDataReferenceEqual(T[] other)
            => IsOnRAM && ReferenceEquals(CPUData, other);

        /// <summary>
        /// Fill the data with the specified value.
        /// </summary>
        /// <param name="value">The value to fill the data with.</param>
        public void Fill(T value)
        {
            if (Location == BufferLocation.Ram)
                Array.Fill(CPUData, value);
            else
                MemoryManager.Fill(AcceleratorData, value);

        }

        /// <summary>
        /// Set all the data to zero.
        /// </summary>
        public void MemSetToZero()
        {
            if (IsOnRAM)
                Array.Clear(CPUData, 0, CPUData.Length);
            else if (IsOnAccelerator)
                AcceleratorData.MemSetToZero();
        }

        public void CopyTo(AcceleratorBuffer<T> destination)
        {
            if (destination.Length != Length)
                throw new ArgumentException($"Expected length {Length}, got {destination.Length}");
            if (IsOnRAM)
                destination.CPUData = CPUData;
            else if (IsOnAccelerator)
                destination.AcceleratorData.CopyToCPU(destination.CPUData);
        }

        public static implicit operator T[](AcceleratorBuffer<T> gpu) => gpu.CPUData;
        public static implicit operator MemoryBuffer1D<T, Stride1D.Dense>(AcceleratorBuffer<T> gpu) => gpu.AcceleratorData;
        public static explicit operator T(AcceleratorBuffer<T> gpu) => gpu.Length == 1 ? gpu.CPUData[0] : throw new InvalidCastException($"Cannot cast a buffer of length {gpu.Length} to a scalar.");
    }

    public class AcceleratorBufferReal<T> : AcceleratorBuffer<T>
        where T : unmanaged, IFloatingPoint<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
    {
        public AcceleratorBufferReal(MemoryManagementUnit mmu, long length) : base(mmu, length) { }

        public AcceleratorBufferReal(MemoryManagementUnit mmu, T[] data) : base(mmu, data) { }

        public AcceleratorBufferReal(MemoryManagementUnit mmu, MemoryBuffer1D<T, Stride1D.Dense> data) : base(mmu, data) { }
    }
}